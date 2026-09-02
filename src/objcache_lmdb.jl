"""
    ObjCache.LMDBFallback

Fallback store for `CompilerCaching.ObjCache` on Julia builds without the runtime's
key/value object cache API. Only included on those builds; on Julia with the runtime
API the runtime's decision about the store is final and LMDB is not even loaded.

The store is a port of the runtime's `src/objcache.cpp` (as of Julia 1.14.0-DEV.3075),
so consumers see the same behaviour on every Julia version:

- **Layout**: one LMDB env with two named databases. `objcache` maps
  `O\\0<hash>` to the value; `objmeta` maps `O\\0<hash>` to its `Int64` access time and
  holds an empty-valued `M\\0<big-endian time><hash>` index entry per object, so the
  least recently used entries are a range scan. `hash` is the runtime's
  `SHA1("JLKV\\0" ‖ u64be(length(ns)) ‖ ns ‖ key)`. `objmeta["schema"]` records the
  layout version; a mismatch disables the store rather than wiping it.
- **Configuration**: `JULIA_OBJCACHE=0` disables the store; `JULIA_OBJCACHE_CAPACITY`
  (bytes) is the budget, default 512 MiB on 64-bit; the env's map is twice the budget.
  Location is `\$JULIA_OBJCACHE_PATH/fallback-lmdb1` or a Julia-minor-specific directory
  in the `objcache-fallback` scratch space of CompilerCaching. Like the runtime, the
  store is off under rr and when the env cannot be opened (network filesystem,
  pid-namespace mismatch, permissions).
- **Writes** overwrite. Values whose page estimate exceeds
  `EVICT_MAX_ENTRY_PAGES` (≈32 MiB) are dropped silently. Before a write, if live pages
  of both databases plus the new value exceed the budget, least recently used entries
  are evicted until usage is below 3/4 of it, in transactions bounded by
  `EVICT_ENTRY_BUDGET` entries and `EVICT_PAGE_BUDGET` pages. Access times have
  one-second resolution; hits set bit 62 so entries that were never read are evicted
  before entries that were read at least once, and a hit refreshes the time only when it
  is more than `ATIME_GRANULARITY` seconds newer.
- **Reads** copy the value out of the map and refresh its access time.

The one deliberate difference: writes are synchronous. The runtime queues them for a
writer thread and drops the queue at exit; here `put!` returns once the entry is
committed, which is strictly stronger and lets tests skip polling.

Every LMDB failure is downgraded to a `@debug` message plus a miss or `false`: a cache
must never fail a compile.
"""
module LMDBFallback

import ..ObjCache
import ...CompilerCaching
import LMDB
using SHA: sha1
using Scratch: get_scratch!

# ===========================================================================
# Constants (objcache.cpp)
# ===========================================================================

const SCHEMA = Int32(1)
const DEFAULT_CAPACITY = Sys.WORD_SIZE == 64 ? (512 << 20) : (32 << 20)
const EVICT_PAGE_BUDGET = 128
const EVICT_ENTRY_BUDGET = 64
const EVICT_MAX_ENTRY_PAGES = 8192
# Seconds. Skip an atime refresh when the stored time is within this window.
const ATIME_GRANULARITY = Int64(300)
# Set on the access time of entries that were read at least once, so never-read
# entries sort (and evict) first.
const HIT_BIT = Int64(1) << 62

const HASH_BYTES = 20
const OBJKEY_SIZE = 2 + HASH_BYTES
const METAKEY_SIZE = 2 + 8 + HASH_BYTES
const OBJKEY_TAG = UInt8('O')
const METAKEY_TAG = UInt8('M')

page_estimate(bytes::Integer, psize::Integer) = bytes ÷ psize + 4

# ===========================================================================
# Keys
# ===========================================================================

"The runtime's KV hash: `SHA1(\"JLKV\\0\" ‖ u64be(length(ns)) ‖ ns ‖ key)`."
function entry_hash(ns::AbstractString, key::AbstractVector{UInt8})
    data = UInt8[0x4a, 0x4c, 0x4b, 0x56, 0x00]   # "JLKV\0"
    n = UInt64(ncodeunits(ns))
    for i in 7:-1:0
        push!(data, UInt8((n >> (8i)) & 0xff))
    end
    append!(data, codeunits(ns))
    append!(data, key)
    return sha1(data)
end

function objkey(hash::AbstractVector{UInt8})
    out = UInt8[OBJKEY_TAG, 0x00]
    append!(out, hash)
    return out
end

function metakey(time::Int64, hash::AbstractVector{UInt8})
    out = UInt8[METAKEY_TAG, 0x00]
    u = reinterpret(UInt64, time)
    for i in 7:-1:0
        push!(out, UInt8((u >> (8i)) & 0xff))
    end
    append!(out, hash)
    return out
end

is_metakey(key::AbstractVector{UInt8}) =
    length(key) == METAKEY_SIZE && key[1] == METAKEY_TAG && key[2] == 0x00

function metakey_parts(key::AbstractVector{UInt8})
    u = UInt64(0)
    for i in 3:10
        u = (u << 8) | UInt64(key[i])
    end
    return reinterpret(Int64, u), key[11:end]
end

# ===========================================================================
# Cache handle
# ===========================================================================

"""
    Cache

An opened store. `clock` returns the current time in seconds as `Int64`; tests inject
a fake clock since the runtime's one-second resolution makes ordering within a second
arbitrary.
"""
mutable struct Cache
    env::LMDB.Environment
    objcache::LMDB.Database
    objmeta::LMDB.Database
    psize::Int
    capacity::Int
    path::String
    clock::Any
    lock::ReentrantLock     # serializes write transactions
end

isopen(cache::Cache) = LMDB.isopen(cache.env)
now_seconds() = Int64(floor(time()))

# LMDB.jl does not guard against use after `close`; a transaction on a closed env faults.
checkopen(cache::Cache) = isopen(cache) || error("object cache store is closed")

struct SchemaMismatch <: Exception
    found::Int32
end
Base.showerror(io::IO, e::SchemaMismatch) =
    print(io, "object cache schema mismatch: found $(e.found), expected $SCHEMA")

"""
    open(path; capacity=DEFAULT_CAPACITY, clock=now_seconds) -> Cache

Open or create the store at `path`, as the runtime does: map size twice the capacity,
`MDB_NOSYNC | MDB_NOTLS`, mode `0o640`, databases `objcache` and `objmeta`, and a
`schema` record whose mismatch throws [`SchemaMismatch`](@ref).
"""
function open(path::AbstractString; capacity::Integer=DEFAULT_CAPACITY, clock=now_seconds)
    mkpath(path)
    env = LMDB.Environment(path; mapsize=2 * Csize_t(capacity), maxreaders=510, maxdbs=128,
                           flags=LMDB.MDB_NOSYNC | LMDB.MDB_NOTLS, mode=0o640)
    try
        objcache, objmeta, psize = LMDB.Transaction(env) do txn
            objcache = LMDB.Database(txn, "objcache"; flags=LMDB.MDB_CREATE)
            objmeta = LMDB.Database(txn, "objmeta"; flags=LMDB.MDB_CREATE)
            psize = LMDB.stat(txn, objcache).psize
            found = try
                LMDB.put!(txn, objmeta, "schema", SCHEMA; flags=LMDB.MDB_NOOVERWRITE)
                SCHEMA
            catch err
                (err isa LMDB.LMDBError && err.code == LMDB.MDB_KEYEXIST) || rethrow()
                LMDB.get(txn, objmeta, "schema", Int32)
            end
            found == SCHEMA || throw(SchemaMismatch(found))
            (objcache, objmeta, psize)
        end
        return Cache(env, objcache, objmeta, Int(psize), Int(capacity), String(path),
                     clock, ReentrantLock())
    catch
        LMDB.close(env)
        rethrow()
    end
end

"Release the environment. Idempotent."
function close(cache::Cache)
    LMDB.isopen(cache.env) || return
    LMDB.close(cache.env, cache.objcache)
    LMDB.close(cache.env, cache.objmeta)
    LMDB.close(cache.env)
    return
end

dbsize(txn, dbi, psize) = let s = LMDB.stat(txn, dbi)
    (s.branch_pages + s.leaf_pages + s.overflow_pages) * psize
end

"Live bytes of both databases, the quantity the runtime compares with the capacity."
used_bytes(cache::Cache, txn) =
    dbsize(txn, cache.objcache, cache.psize) + dbsize(txn, cache.objmeta, cache.psize)
used_bytes(cache::Cache) = (checkopen(cache);
    LMDB.Transaction(cache.env; flags=LMDB.MDB_RDONLY) do txn
        used_bytes(cache, txn)
    end)

"Number of stored objects."
entries(cache::Cache) = (checkopen(cache);
    LMDB.Transaction(cache.env; flags=LMDB.MDB_RDONLY) do txn
        LMDB.stat(txn, cache.objcache).entries
    end)

# ===========================================================================
# get / put!
# ===========================================================================

"""
    get(cache, ns, key) -> Union{Vector{UInt8}, Nothing}

Copy of the value stored for `(ns, key)`, or `nothing`. A hit refreshes the entry's
access time (with the hit bit set); failures of that refresh are ignored.
"""
function get(cache::Cache, ns::AbstractString, key::AbstractVector{UInt8})
    checkopen(cache)
    hash = entry_hash(ns, key)
    ok = objkey(hash)
    blob = LMDB.Transaction(cache.env; flags=LMDB.MDB_RDONLY) do txn
        LMDB.get(txn, cache.objcache, ok, Vector{UInt8}, nothing)
    end
    blob === nothing && return nothing
    try
        Base.@lock cache.lock LMDB.Transaction(cache.env) do txn
            update_atime!(cache, txn, hash, cache.clock() | HIT_BIT, false)
        end
    catch err
        @debug "ObjCache fallback: access time refresh failed" exception=(err, catch_backtrace())
    end
    return blob
end

"""
    put!(cache, ns, key, value) -> Bool

Store `value` for `(ns, key)`, overwriting any existing entry, after evicting least
recently used entries if the budget would be exceeded. Returns `true` when the entry was
committed **or dropped for being too large** (the runtime reports the latter as queued
too); `false` when the write failed.
"""
function put!(cache::Cache, ns::AbstractString, key::AbstractVector{UInt8},
              value::AbstractVector{UInt8})
    checkopen(cache)
    page_estimate(length(value), cache.psize) > EVICT_MAX_ENTRY_PAGES && return true
    hash = entry_hash(ns, key)
    Base.@lock cache.lock begin
        maybe_evict_lru!(cache, length(value)) || return false
        try
            LMDB.Transaction(cache.env) do txn
                LMDB.put!(txn, cache.objcache, objkey(hash), convert(Vector{UInt8}, value))
                update_atime!(cache, txn, hash, cache.clock(), true)
            end
        catch err
            (err isa LMDB.LMDBError && err.code in (LMDB.MDB_MAP_FULL, LMDB.MDB_TXN_FULL)) || rethrow()
            @debug "ObjCache fallback: store full, entry skipped" exception=err
            return false
        end
    end
    return true
end

# `objcache.cpp: ObjCache::updateATime`. `fresh` marks a write-time update, which must
# create the record; a hit on an entry that was evicted meanwhile is a no-op.
function update_atime!(cache::Cache, txn, hash::Vector{UInt8}, time::Int64, fresh::Bool)
    ok = objkey(hash)
    old = LMDB.get(txn, cache.objmeta, ok, Int64, nothing)
    if old === nothing
        fresh || return
    else
        time < old + ATIME_GRANULARITY && return
        LMDB.delete!(txn, cache.objmeta, metakey(old, hash))
    end
    LMDB.put!(txn, cache.objmeta, ok, time)
    LMDB.put!(txn, cache.objmeta, metakey(time, hash), UInt8[])
    return
end

# ===========================================================================
# Eviction (objcache.cpp: ObjCache::maybeEvictLRU)
# ===========================================================================

"""
    maybe_evict_lru!(cache, room_for) -> Bool

If live usage plus `room_for` bytes exceeds the capacity, delete least recently used
entries until usage drops below 3/4 of it. Each transaction removes at most
`EVICT_ENTRY_BUDGET` entries or `EVICT_PAGE_BUDGET` pages; a full-map failure halves the
batch and retries down to single entries. Returns whether the store has room.
"""
function maybe_evict_lru!(cache::Cache, room_for::Integer)
    room = cld(room_for, cache.psize) * cache.psize
    used = used_bytes(cache) + room
    used <= cache.capacity && return true
    threshold = cache.capacity * 3 ÷ 4

    entry_limit = EVICT_ENTRY_BUDGET
    while used_bytes(cache) + room > threshold
        evicted = try
            evict_batch!(cache, room, threshold, entry_limit)
        catch err
            (err isa LMDB.LMDBError && err.code in (LMDB.MDB_MAP_FULL, LMDB.MDB_TXN_FULL)) || rethrow()
            entry_limit == 1 && return false
            entry_limit = max(1, entry_limit ÷ 2)
            continue
        end
        evicted == 0 && return false
        entry_limit = EVICT_ENTRY_BUDGET
    end
    return true
end

# One eviction transaction. Returns the number of entries removed.
function evict_batch!(cache::Cache, room::Integer, threshold::Integer, entry_limit::Integer)
    LMDB.Transaction(cache.env) do txn
        evicted = 0
        pages = 0
        LMDB.Cursor(txn, cache.objmeta) do cur
            key = LMDB.seek_range!(cur, metakey(Int64(0), zeros(UInt8, HASH_BYTES)))
            while key !== nothing && is_metakey(key) &&
                  used_bytes(cache, txn) + room > threshold
                _, hash = metakey_parts(key)
                ok = objkey(hash)
                value = LMDB.get(txn, cache.objcache, ok, Vector{UInt8}, nothing)
                entry_pages = value === nothing ? 4 : page_estimate(length(value), cache.psize)
                if entry_pages > EVICT_MAX_ENTRY_PAGES
                    key = LMDB.next!(cur)
                    continue
                end
                # Bound the transaction; oversized first entries go alone.
                evicted > 0 && (evicted >= entry_limit ||
                                entry_pages > EVICT_PAGE_BUDGET - pages) && break
                LMDB.delete!(txn, cache.objcache, ok)
                LMDB.delete!(txn, cache.objmeta, ok)
                LMDB.delete!(cur)
                pages += entry_pages
                evicted += 1
                entry_pages > EVICT_PAGE_BUDGET && break
                key = LMDB.next!(cur)
            end
        end
        evicted
    end
end

# ===========================================================================
# Configuration (mirrors the runtime's environment variables)
# ===========================================================================

"Whether `JULIA_OBJCACHE=0` opts out of the store."
configured_enabled(env=ENV) = Base.get(env, "JULIA_OBJCACHE", "") != "0"

"""
    configured_capacity(env=ENV) -> Int

`JULIA_OBJCACHE_CAPACITY` in bytes (decimal, or `0x`/`0o`/`0b` prefixed), else
[`DEFAULT_CAPACITY`](@ref). Invalid values are reported and ignored, like the runtime.
"""
function configured_capacity(env=ENV)
    setting = Base.get(env, "JULIA_OBJCACHE_CAPACITY", "")
    isempty(setting) && return DEFAULT_CAPACITY
    value = tryparse(UInt64, setting)
    if value === nothing || value == 0 || value > typemax(Int) ÷ 2
        @warn "objcache: invalid value for JULIA_OBJCACHE_CAPACITY: $setting"
        return DEFAULT_CAPACITY
    end
    return Int(value)
end

"""
    configured_path(env=ENV) -> String

`\$JULIA_OBJCACHE_PATH/fallback-lmdb1` when the variable is set (a subdirectory, so the
same setting never points the runtime store and this one at the same env), else the
`objcache-fallback/v<major>.<minor>` in CompilerCaching's scratch space.
"""
function configured_path(env=ENV)
    override = Base.get(env, "JULIA_OBJCACHE_PATH", "")
    isempty(override) || return joinpath(abspath(expanduser(override)), "fallback-lmdb1")
    root = get_scratch!(CompilerCaching, "objcache-fallback")
    return joinpath(root, "v$(VERSION.major).$(VERSION.minor)")
end

# rr sets this in the tracee environment; the runtime symbol is not exported on 1.12.
running_under_rr(env=ENV) = haskey(env, "RUNNING_UNDER_RR")

# ===========================================================================
# ObjCache backend
# ===========================================================================

"""
    LMDBStore

The `ObjCache.AbstractStore` backed by this module. Opens the env lazily on first use;
a failed open disables the store for the rest of the process.
"""
mutable struct LMDBStore <: ObjCache.AbstractStore
    lock::ReentrantLock
    initialized::Bool
    cache::Union{Cache,Nothing}
    LMDBStore() = new(ReentrantLock(), false, nothing)
end

function cache(store::LMDBStore)
    Base.@lock store.lock begin
        if !store.initialized
            store.cache = try_init(store)
            store.initialized = true
        end
        return store.cache
    end
end

function try_init(::LMDBStore)
    configured_enabled() || return nothing
    running_under_rr() && return nothing
    try
        return open(configured_path(); capacity=configured_capacity())
    catch err
        @debug "ObjCache fallback store disabled" exception=(err, catch_backtrace())
        return nothing
    end
end

ObjCache.store_enabled(store::LMDBStore) = cache(store) !== nothing

function ObjCache.store_get(store::LMDBStore, ns::String, key::Vector{UInt8})
    c = cache(store)
    c === nothing && return nothing
    try
        return get(c, ns, key)
    catch err
        @debug "ObjCache fallback lookup failed" exception=(err, catch_backtrace())
        return nothing
    end
end

function ObjCache.store_put!(store::LMDBStore, ns::String, key::Vector{UInt8}, val::Vector{UInt8})
    c = cache(store)
    c === nothing && return false
    try
        return put!(c, ns, key, val)
    catch err
        @debug "ObjCache fallback store failed" exception=(err, catch_backtrace())
        return false
    end
end

"The process-wide fallback store."
const STORE = LMDBStore()

end # module LMDBFallback
