"""Codificador, decodificador y motor de búsqueda para chunks STDT (StringTableData) de FMOD Studio 1.08.

Automobilista 2 resuelve las rutas de eventos de los archivos .csdbin exclusivamente mediante
`FMOD::Studio::System::lookupID(path, &guid)`, el cual recorre las tablas RadixTree (`STDT`) de todos
los bancos `.bank` cargados en memoria. Incrustar un chunk `STDT` válido dentro de nuestro `.bank`
mod permite usar rutas limpias `event:/AMS2/Vehicles/Mods/<mod>/...` en el `.csdbin` sin tocar
jamás `MasterBank.strings.bank`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


def _ascii_lower_byte(b: int) -> int:
    """Réplica exacta del plegado ASCII de FMOD 1.08 (`if ('A' <= b <= 'Z') b += 0x20`)."""
    return b + 0x20 if 0x41 <= b <= 0x5A else b


def _ascii_lower_bytes(data: bytes) -> bytes:
    return bytes(_ascii_lower_byte(b) for b in data)


def _guid_struct_sort_key(guid_bytes: bytes) -> tuple[int, int, int, bytes]:
    """Orden `FMOD_GUID::operator<` usado por `StringTable::getPath` en el arreglo `mGuids` de STDT."""
    d1, d2, d3 = struct.unpack_from("<IHH", guid_bytes, 0)
    return (d1, d2, d3, guid_bytes[8:16])


def pack_fmod_uint(val: int) -> bytes:
    """Codifica un entero sin signo en el formato `PackedUInt` de 16/32 bits de `fmod_serialization.cpp`."""
    if val < 0 or val > 0x7FFFFFFF:
        raise ValueError(f"Valor fuera de rango para FMOD PackedUInt: {val}")
    if val <= 0x7FFF:
        return struct.pack("<H", val)
    return struct.pack("<HH", (val & 0x7FFF) | 0x8000, (val >> 15) & 0xFFFF)


def unpack_fmod_uint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decodifica un `PackedUInt` de FMOD Studio 1.08 retornando `(valor, nueva_posicion)`."""
    w0 = struct.unpack_from("<H", buf, pos)[0]
    pos += 2
    if w0 & 0x8000:
        w1 = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        return ((w0 & 0x7FFF) | (w1 << 15)), pos
    return w0, pos


def _pack_u24_le(val: int) -> bytes:
    return bytes((val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF))


@dataclass
class _TrieNode:
    edge: bytes = b""  # Subcadena UTF-8 en mayúsculas/minúsculas originales (vacía en raíz o nodo terminal nulo)
    guid_idx: int | None = None
    children: dict[int, "_TrieNode"] = field(default_factory=dict)  # first_lower_byte -> _TrieNode


def _common_prefix_len_ci(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and _ascii_lower_byte(a[i]) == _ascii_lower_byte(b[i]):
        i += 1
    return i


def _insert_patricia(root: _TrieNode, path_bytes: bytes, guid_idx: int) -> None:
    node = root
    rem = path_bytes
    while rem:
        key = _ascii_lower_byte(rem[0])
        child = node.children.get(key)
        if child is None:
            node.children[key] = _TrieNode(edge=rem, guid_idx=guid_idx)
            return

        cp = _common_prefix_len_ci(rem, child.edge)
        if cp == len(child.edge):
            rem = rem[cp:]
            node = child
            continue

        # Dividir la arista existente en un nodo intermedio con el prefijo común
        split_node = _TrieNode(edge=child.edge[:cp], guid_idx=None)
        old_suffix = child.edge[cp:]
        child.edge = old_suffix
        split_node.children[_ascii_lower_byte(old_suffix[0])] = child

        new_suffix = rem[cp:]
        if new_suffix:
            split_node.children[_ascii_lower_byte(new_suffix[0])] = _TrieNode(
                edge=new_suffix, guid_idx=guid_idx
            )
        else:
            split_node.guid_idx = guid_idx

        node.children[key] = split_node
        return

    node.guid_idx = guid_idx


def _normalize_terminal_children(node: _TrieNode) -> None:
    """En FMOD 1.08, si un nodo tiene hijos Y además es fin de ruta exacta, usa un hijo nulo (`b3=0x00`)."""
    for ch in list(node.children.values()):
        _normalize_terminal_children(ch)
    if node.children and node.guid_idx is not None:
        node.children[0] = _TrieNode(edge=b"", guid_idx=node.guid_idx)
        node.guid_idx = None


def build_stdt_chunk(entries: list[tuple[str, bytes]]) -> bytes:
    """Construye un payload binario `STDT` (RadixTree tipo 1) 100% compatible con FMOD Studio 1.08.

    `entries`: lista de pares `(path_str, guid_16_bytes)`, p.ej.
      `[("bank:/mod_Engine", b"..."), ("event:/AMS2/Vehicles/Mods/mod/EngineExt", b"...")]`.
    """
    if not entries:
        return b""

    # Deduplicar por ruta case-insensitive y ordenar GUIDs según FMOD_GUID::operator<
    dedup_by_lower: dict[bytes, tuple[str, bytes]] = {}
    for path_str, guid_bytes in entries:
        if len(guid_bytes) != 16:
            raise ValueError(f"GUID inválido para '{path_str}': se esperaban 16 bytes.")
        p_clean = path_str.strip()
        if not p_clean:
            continue
        dedup_by_lower[_ascii_lower_bytes(p_clean.encode("utf-8"))] = (p_clean, bytes(guid_bytes))

    if not dedup_by_lower:
        return b""

    # Construir tabla mGuids ordenada estrictamente por FMOD_GUID::operator<
    unique_guids = sorted({g for _, g in dedup_by_lower.values()}, key=_guid_struct_sort_key)
    guid_to_idx = {g: idx for idx, g in enumerate(unique_guids)}

    # Construir árbol Patricia / Radix
    root = _TrieNode()
    for _, (p_str, g_bytes) in sorted(dedup_by_lower.items(), key=lambda kv: kv[0]):
        _insert_patricia(root, p_str.encode("utf-8"), guid_to_idx[g_bytes])
    _normalize_terminal_children(root)

    # Asignar índices de nodos de forma que los hijos de cada nodo queden contiguos (requisito FMOD)
    flat_nodes: list[_TrieNode] = [root]
    parent_indices: list[int] = [0xFFFFFF]
    first_child_map: dict[int, int] = {}
    num_children_map: dict[int, int] = {}

    q_pos = 0
    while q_pos < len(flat_nodes):
        cur_node = flat_nodes[q_pos]
        if cur_node.children:
            sorted_keys = sorted(cur_node.children.keys())
            if len(sorted_keys) > 255:
                raise ValueError("Un nodo STDT excede el máximo de 255 ramas hijas.")
            first_child_map[q_pos] = len(flat_nodes)
            num_children_map[q_pos] = len(sorted_keys)
            for k in sorted_keys:
                flat_nodes.append(cur_node.children[k])
                parent_indices.append(q_pos)
        else:
            num_children_map[q_pos] = 0
        q_pos += 1

    # Construir pool de subcadenas terminadas en '\0' (mStringData) y nodos empaquetados (mNodes)
    string_pool = bytearray()
    str_offset_cache: dict[bytes, int] = {}
    packed_nodes = bytearray()
    leaf_for_guid: dict[int, int] = {}

    for n_idx, tnode in enumerate(flat_nodes):
        if n_idx == 0 or not tnode.edge:
            b3 = 0x00
            str_off = 0xFFFFFF
        else:
            b3 = _ascii_lower_byte(tnode.edge[0])
            cached = str_offset_cache.get(tnode.edge)
            if cached is None:
                cached = len(string_pool)
                string_pool += tnode.edge + b"\x00"
                str_offset_cache[tnode.edge] = cached
            str_off = cached

        if str_off != 0xFFFFFF and str_off > 0xFFFFFE:
            raise ValueError("Pool de cadenas STDT excede el límite de 24 bits.")

        n_ch = num_children_map[n_idx]
        if n_ch > 0:
            low24 = first_child_map[n_idx]
        else:
            assert tnode.guid_idx is not None
            low24 = tnode.guid_idx
            leaf_for_guid[tnode.guid_idx] = n_idx

        w0 = ((b3 & 0xFF) << 24) | (str_off & 0x00FFFFFF)
        w1 = ((n_ch & 0xFF) << 24) | (low24 & 0x00FFFFFF)
        packed_nodes += struct.pack("<II", w0, w1)

    # Arreglo u24_1 (índice de hoja para cada GUID, usado por EventDescription::getPath / Bank::getPath)
    u24_leaf_bytes = bytearray()
    for g_i in range(len(unique_guids)):
        u24_leaf_bytes += _pack_u24_le(leaf_for_guid.get(g_i, 0xFFFFFF))

    # Arreglo u24_2 (índice de nodo padre para cada nodo)
    u24_parent_bytes = bytearray()
    for p_i in parent_indices:
        u24_parent_bytes += _pack_u24_le(p_i)

    # Ensamblar las 5 secciones exactas de STDT (tipo 1 = RadixTree)
    out = bytearray()
    out += struct.pack("<I", 1)
    # 1. DynamicArray<Node> (con indicador de stride = 8)
    out += pack_fmod_uint((len(flat_nodes) << 1) | 1) + pack_fmod_uint(8) + packed_nodes
    # 2. DynamicArray<FMOD_GUID> (con indicador de stride = 16)
    out += pack_fmod_uint((len(unique_guids) << 1) | 1) + pack_fmod_uint(16) + b"".join(unique_guids)
    # 3. DynamicArray<char>
    out += pack_fmod_uint(len(string_pool)) + string_pool
    # 4. PackedU24 leaf indices (por GUID)
    out += pack_fmod_uint(len(unique_guids)) + u24_leaf_bytes
    # 5. PackedU24 parent indices (por Nodo)
    out += pack_fmod_uint(len(flat_nodes)) + u24_parent_bytes
    return bytes(out)


def parse_stdt_sections(
    stdt: bytes,
) -> tuple[list[tuple[int, int]], list[bytes], bytes, list[int], list[int]]:
    """Desempaqueta las 5 tablas internas de un bloque `STDT` de FMOD Studio 1.08."""
    if len(stdt) < 12:
        return [], [], b"", [], []
    table_type = struct.unpack_from("<I", stdt, 0)[0]
    if table_type != 1:
        raise ValueError(f"Tipo de tabla STDT no soportado: {table_type}")

    pos = 4
    raw_n1, pos = unpack_fmod_uint(stdt, pos)
    stride1, pos = unpack_fmod_uint(stdt, pos) if (raw_n1 & 1) else (8, pos)
    n_nodes = raw_n1 >> 1
    nodes = [struct.unpack_from("<II", stdt, pos + i * stride1) for i in range(n_nodes)]
    pos += n_nodes * stride1

    raw_n2, pos = unpack_fmod_uint(stdt, pos)
    stride2, pos = unpack_fmod_uint(stdt, pos) if (raw_n2 & 1) else (16, pos)
    n_guids = raw_n2 >> 1
    guids = [bytes(stdt[pos + i * stride2 : pos + i * stride2 + 16]) for i in range(n_guids)]
    pos += n_guids * stride2

    n_chars, pos = unpack_fmod_uint(stdt, pos)
    chars = bytes(stdt[pos : pos + n_chars])
    pos += n_chars

    n_u24_1, pos = unpack_fmod_uint(stdt, pos)
    u24_1 = [
        stdt[pos + i * 3] | (stdt[pos + i * 3 + 1] << 8) | (stdt[pos + i * 3 + 2] << 16)
        for i in range(n_u24_1)
    ]
    pos += n_u24_1 * 3

    n_u24_2, pos = unpack_fmod_uint(stdt, pos)
    u24_2 = [
        stdt[pos + i * 3] | (stdt[pos + i * 3 + 1] << 8) | (stdt[pos + i * 3 + 2] << 16)
        for i in range(n_u24_2)
    ]
    return nodes, guids, chars, u24_1, u24_2


def lookup_stdt_id(
    stdt: bytes,
    path: str,
    _parsed: tuple[list[tuple[int, int]], list[bytes], bytes, list[int], list[int]] | None = None,
) -> bytes | None:
    """Réplica exacta instrucción por instrucción de `FMOD::Studio::StringTable::lookupID` en `AMS2AVX.exe`."""
    if not stdt and _parsed is None:
        return None
    nodes, guids, chars, _, _ = _parsed if _parsed is not None else parse_stdt_sections(stdt)
    if not nodes or not guids or not chars:
        return None

    q = path.encode("utf-8")
    q_pos = 0
    cur_idx = 0

    while q_pos < len(q):
        _, w1 = nodes[cur_idx]
        num_ch = (w1 >> 24) & 0xFF
        if num_ch == 0:
            return None
        first_ch = w1 & 0x00FFFFFF
        target_b3 = _ascii_lower_byte(q[q_pos])

        matched_child: int | None = None
        for i in range(num_ch):
            cand_idx = first_ch + i
            cw0, _ = nodes[cand_idx]
            if ((cw0 >> 24) & 0xFF) == target_b3:
                matched_child = cand_idx
                break
        if matched_child is None:
            return None

        cw0, _ = nodes[matched_child]
        str_off = cw0 & 0x00FFFFFF
        q_pos += 1
        s_pos = str_off + 1

        while s_pos < len(chars) and chars[s_pos] != 0:
            if q_pos >= len(q):
                return None
            if _ascii_lower_byte(q[q_pos]) != _ascii_lower_byte(chars[s_pos]):
                return None
            q_pos += 1
            s_pos += 1

        cur_idx = matched_child

    # Fin de la cadena consultada: verificar si el nodo actual es hoja o tiene hijo terminal nulo (b3=0)
    _, w1 = nodes[cur_idx]
    num_ch = (w1 >> 24) & 0xFF
    if num_ch != 0:
        first_ch = w1 & 0x00FFFFFF
        null_child: int | None = None
        for i in range(num_ch):
            cand_idx = first_ch + i
            cw0, _ = nodes[cand_idx]
            if ((cw0 >> 24) & 0xFF) == 0:
                null_child = cand_idx
                break
        if null_child is None:
            return None
        cur_idx = null_child
        _, w1 = nodes[cur_idx]
        if ((w1 >> 24) & 0xFF) != 0:
            return None

    guid_idx = w1 & 0x00FFFFFF
    if guid_idx >= len(guids):
        return None
    return guids[guid_idx]


def parse_stdt_all_paths(stdt: bytes) -> dict[bytes, str]:
    """Reconstruye el mapa `{guid_bytes: path_str}` desde un chunk `STDT` (equivalente a `getPath`)."""
    if not stdt:
        return {}
    nodes, guids, chars, u24_1, u24_2 = parse_stdt_sections(stdt)
    result: dict[bytes, str] = {}
    for g_idx, g_bytes in enumerate(guids):
        if g_idx >= len(u24_1):
            break
        cur = u24_1[g_idx]
        if cur >= len(nodes):
            continue
        parts: list[str] = []
        while cur != 0 and cur != 0xFFFFFF and cur < len(nodes):
            w0, _ = nodes[cur]
            s_off = w0 & 0x00FFFFFF
            if s_off != 0xFFFFFF and s_off < len(chars):
                end = chars.find(b"\x00", s_off)
                if end >= 0:
                    parts.append(chars[s_off:end].decode("utf-8", "replace"))
            cur = u24_2[cur] if cur < len(u24_2) else 0xFFFFFF
        result[g_bytes] = "".join(reversed(parts))
    return result
