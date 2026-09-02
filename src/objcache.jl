"""
    CompilerCaching.ObjCache

Key/value access to Julia's persistent object cache, for storing compiled artifacts
across sessions.

Compilation results managed by CompilerCaching live in three tiers:

1. **Session**: the results struct attached to a `CodeInstance` (see `results`).
2. **Package image**: the same struct, serialized during precompilation.
3. **Object cache** (this module): opaque byte artifacts produced at run time, stored
   in the runtime's on-disk object cache (Julia ≥ 1.14.0-DEV.3075) or, on older Julia,
   in an LMDB-backed fallback store owned by this package (`ObjCache.LMDBFallback`).

This module owns mechanism only: it does not know what an artifact depends on, nor
whether it is safe to persist. Consumers (compiler drivers) own that policy. The
contract every consumer must honour is documented in [`get!`](@ref).
"""
module ObjCache

using SHA: SHA256_CTX, update!, digest!

import ..CompilerCaching: @public

@public enabled, get, put!, keyhash, get!

"Whether this Julia build exposes the runtime's key/value object cache API."
const HAS_KV_API = VERSION >= v"1.14.0-DEV.3075"


#==============================================================================#
# Backends
#==============================================================================#

"""
    AbstractStore

A persistent byte store. Backends implement `store_enabled`, `store_get` and
`store_put!`. Keys and values are `Vector{UInt8}`; namespaces are `String`.
"""
abstract type AbstractStore end

function store_enabled end
function store_get end
function store_put! end

"The runtime's object cache (`src/objcache.cpp`), available when `HAS_KV_API`."
struct RuntimeStore <: AbstractStore end

@static if HAS_KV_API
    store_enabled(::RuntimeStore) =
        ccall(:jl_objcache_kv_enabled, Cint, ()) != 0
    store_get(::RuntimeStore, ns::String, key::Vector{UInt8}) =
        ccall(:jl_objcache_kv_get, Any, (Cstring, Ptr{UInt8}, Csize_t),
              ns, key, length(key))::Union{Vector{UInt8},Nothing}
    store_put!(::RuntimeStore, ns::String, key::Vector{UInt8}, val::Vector{UInt8}) =
        ccall(:jl_objcache_kv_put, Cint, (Cstring, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t),
              ns, key, length(key), val, length(val)) != 0
end

# On Julia without the KV API, an LMDB-backed store owned by this package takes the
# runtime's place. Only loaded there: on 1.14+ the runtime's decision is final and no
# second store is opened next to it (nor is LMDB loaded at all).
@static if !HAS_KV_API
    include("objcache_lmdb.jl")
end

# Per-process memo of the availability probe.
const enabled_state = Ref{Union{Nothing,Bool}}(nothing)

"The active backend. Does not probe availability."
store() = @static HAS_KV_API ? RuntimeStore() : LMDBFallback.STORE


#==============================================================================#
# Availability
#==============================================================================#

"""
    enabled() -> Bool

Whether a persistent store is available in this process. The answer is probed once
and memoized. The first call may initialize the store (create its directory, open the
environment, and, for the runtime backend, start the writer thread), so call it lazily
on first use rather than from a package `__init__`.

The store is disabled when `JULIA_OBJCACHE=0` is set or when the cache directory is
unusable.
"""
function enabled()
    state = enabled_state[]
    state !== nothing && return state
    result = store_enabled(store())::Bool
    enabled_state[] = result
    return result
end

#==============================================================================#
# Raw access
#==============================================================================#

function checked_namespace(ns::AbstractString)
    str = String(ns)
    occursin('\0', str) &&
        throw(ArgumentError("object-cache namespaces must not contain NUL bytes"))
    return str
end

"""
    get(ns::AbstractString, key::AbstractVector{UInt8}) -> Union{Vector{UInt8},Nothing}

Look up `key` in namespace `ns`. Returns a copy of the stored bytes, or `nothing` on a
miss **or** when no store is available (see [`enabled`](@ref)). A hit refreshes the
entry's position in the LRU order.

`ns` must not contain NUL bytes. See [`get!`](@ref) for the namespace convention.
"""
function get(ns::AbstractString, key::AbstractVector{UInt8})
    ns = checked_namespace(ns)
    enabled() || return nothing
    return store_get(store(), ns, convert(Vector{UInt8}, key))
end

"""
    put!(ns::AbstractString, key::AbstractVector{UInt8}, val::AbstractVector{UInt8}) -> Bool

Store `val` under `key` in namespace `ns`. Returns whether the backend accepted the
write; `false` when no store is available.

Writes are **best-effort**. The runtime backend is asynchronous and does not drain its
queue at exit, so a process that stores an entry and exits immediately may lose it, and
a `get` right after `put!` may still miss. The fallback backend commits synchronously,
but callers must rely only on the weaker runtime contract. Keys are write-once by
contract: the store may overwrite, but callers must never rely on a second `put!` to
update an entry. Entries above roughly 32 MiB are dropped silently.
"""
function put!(ns::AbstractString, key::AbstractVector{UInt8}, val::AbstractVector{UInt8})
    ns = checked_namespace(ns)
    enabled() || return false
    return store_put!(store(), ns, convert(Vector{UInt8}, key),
                      convert(Vector{UInt8}, val))::Bool
end


#==============================================================================#
# Keys
#==============================================================================#

keybytes(x::AbstractVector{UInt8}) = x
keybytes(x::AbstractString) = codeunits(x)
keybytes(x::Integer) = codeunits(string(x))
keybytes(x::VersionNumber) = codeunits(string(x))
keybytes(@nospecialize(x)) = throw(ArgumentError(
    "unsupported key field of type $(typeof(x)); pass bytes, a string, an integer or a VersionNumber"))

function update_u32be!(ctx, x::Integer)
    0 <= x <= typemax(UInt32) ||
        throw(ArgumentError("key framing integer out of range: $x"))
    u = UInt32(x)
    update!(ctx, (UInt8(u >> 24), UInt8((u >> 16) & 0xff), UInt8((u >> 8) & 0xff), UInt8(u & 0xff)))
    return ctx
end

"""
    keyhash(schema::Integer, fields...) -> Vector{UInt8}

Derive a 32-byte key: the SHA-256 of `u32be(schema)` followed by, for every field,
`u32be(length) ‖ bytes`. Fields may be byte vectors, strings (their code units),
integers and `VersionNumber`s (both in canonical decimal/string form). The length
prefix makes the framing injective, so `("ab", "c")` and `("a", "bc")` hash
differently. `schema` is a consumer-chosen version of the value layout; bump it to
invalidate every existing entry.

Fields longer than `typemax(UInt32)` bytes are rejected.
"""
function keyhash(schema::Integer, fields...)
    ctx = SHA256_CTX()
    update_u32be!(ctx, schema)
    for field in fields
        bytes = keybytes(field)
        update_u32be!(ctx, length(bytes))
        update!(ctx, bytes)
    end
    return digest!(ctx)
end

#==============================================================================#
# Read-through helper
#==============================================================================#

"""
    get!(compile, ns::AbstractString, keyfields...; schema::Integer, persistable::Bool) -> Vector{UInt8}
    get!(compile, ns::AbstractString, key::AbstractVector{UInt8}; persistable::Bool) -> Vector{UInt8}

Read-through lookup. Derive the key with [`keyhash`](@ref)`(schema, keyfields...)` (or
use `key` as is), return the stored bytes on a hit, else call
`compile()::AbstractVector{UInt8}`, submit the result for storage, and return it. With
`persistable=false`, or when no store is available, the store is never consulted and
this is plain `compile()`. Exceptions from `compile` propagate; nothing is stored.

`persistable` has no default so that drivers thread their portability trait through
by construction. Pass `false` for artifacts that embed session-local state (addresses,
handles), or whose cross-process compatibility cannot be represented completely by the
key fields.

The single-byte-vector positional form is the raw-key form. Passing `schema` along with
a single byte-vector argument hashes it as a one-field key instead.

# Contract

1. **`ns` and the key fields together cover every input** that influences the value,
   including the identity of any external toolchain (compiler version string, flags).
   Under-keying silently serves stale artifacts.
2. Entries are **write-once**: the same key must always map to equivalent bytes.
3. Writes are **asynchronous and best-effort** (see [`put!`](@ref)): a process that
   compiles and exits at once may not populate the store, and a hit right after a
   miss in the same session is not guaranteed.
4. The store is **per Julia minor version**, not per build. This is only a storage
   partition; it does not make an under-keyed artifact valid across builds.
5. **Keep the returned artifact in your session-tier results struct.** `get!` has no
   in-memory memo; the store is a cross-session tier only.
6. Values above roughly **32 MiB** are dropped silently.
7. The capacity is **shared with the JIT's object cache** (`JULIA_OBJCACHE_CAPACITY`,
   default 512 MiB) with LRU eviction, so an entry may disappear at any time.

Namespace convention: `"<PackageName>/<artifact-kind>"`, for example `"cuTile/cubin"`.
"""
function get!(compile, ns::AbstractString, key::AbstractVector{UInt8};
              schema::Union{Integer,Nothing}=nothing, persistable::Bool)
    if schema !== nothing
        # A single byte-vector field with an explicit schema: the keyfields form.
        persistable && enabled() || return compiled_bytes(compile)
        return get!(compile, ns, keyhash(schema, key); persistable)
    end
    persistable && enabled() || return compiled_bytes(compile)
    rawkey = convert(Vector{UInt8}, key)
    hit = get(ns, rawkey)
    hit !== nothing && return hit
    bytes = compiled_bytes(compile)
    put!(ns, rawkey, bytes)
    return bytes
end

function get!(compile, ns::AbstractString, keyfields...; schema::Integer, persistable::Bool)
    persistable && enabled() || return compiled_bytes(compile)
    return get!(compile, ns, keyhash(schema, keyfields...); persistable)
end

function compiled_bytes(compile)
    val = compile()
    val isa AbstractVector{UInt8} || throw(ArgumentError(
        "ObjCache.get!: compile must return an AbstractVector{UInt8}, got $(typeof(val))"))
    return convert(Vector{UInt8}, val)
end

end # module ObjCache
