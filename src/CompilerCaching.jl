# CompilerCaching.jl - Reusable package for compiler result caching
#
# Leverages Julia's Method/MethodInstance/CodeInstance infrastructure to provide:
# - Lazy compilation with caching
# - Type-based specialization and dispatch
# - Automatic invalidation when methods are redefined
# - Transitive dependency tracking
#
# Requires Julia 1.11+

module CompilerCaching

using Base.Experimental: @MethodTable
using Scratch: @get_scratch!
using Serialization: serialize, deserialize
const CC = Core.Compiler

include("utils.jl")

export CompilerCache, CacheOwner
export add_method, cached_compilation, method_instance
export cache_owner, cache!
export populate!, StackedMethodTable
export disk_cache_path, clear_disk_cache!

#==============================================================================#
# CacheOwner - identifies a cache partition
#==============================================================================#

"""
    CacheOwner{K}

Identifies a cache partition. Constructed internally from a tag and optional keys.

- `tag::Symbol` - Base identifier (e.g., `:SynchJulia`, `:GPUCompiler`)
- `keys::K` - Additional sharding keys (e.g., device capability, optimization level)
"""
struct CacheOwner{K}
    tag::Symbol
    keys::K
end

function Base.hash(o::CacheOwner, h::UInt)
    h = hash(o.tag, h)
    h = hash(o.keys, h)
    return h
end

Base.:(==)(a::CacheOwner, b::CacheOwner) = a.tag == b.tag && a.keys == b.keys

#==============================================================================#
# DiskCacheEntry - serializable cache entry
#==============================================================================#

"""
    DiskCacheEntry{K}

A serializable entry for disk caching.

- `spec_types::Type` - mi.specTypes for validation
- `keys::K` - sharding keys
- `data::Any` - serializable compile result
"""
struct DiskCacheEntry{K}
    spec_types::Type
    keys::K
    data::Any
end

#==============================================================================#
# CompilerCache - main entry point
#==============================================================================#

"""
    CompilerCache{K}

A compilation cache instance, parameterized by key type K.

- `tag::Symbol` - Base tag for cache owner
- `method_table::Union{Core.MethodTable, Nothing}` - Method table for dispatch (nothing = global MT)

# Usage Modes

1. **Custom IR mode**: `CompilerCache(:Tag, MY_MT)` with `add_method` for custom source
2. **Overlay mode**: `CompilerCache(:Tag, MY_MT)` with Julia methods in custom MT
3. **Global mode**: `CompilerCache(:Tag)` with standard Julia methods

# Examples

```julia
using CompilerCaching: CompilerCache

# Mode 1/2: Custom MT (for custom IR or overlay methods)
const synch = CompilerCache(:SynchJulia, MY_MT)

# Mode 3: Global MT (for standard Julia methods)
const gpu = CompilerCache(:GPUCompiler)

# With sharding keys
const gpu = CompilerCache{@NamedTuple{cap::VersionNumber}}(:GPUCompiler)
```
"""
struct CompilerCache{K}
    tag::Symbol
    method_table::Union{Core.MethodTable, Nothing}
    external_cache::Dict{Tuple{Core.CodeInstance, K}, Any}
    lock::ReentrantLock
    disk_cache::Bool
end

# Constructors
function CompilerCache{K}(tag::Symbol, method_table=nothing; disk_cache::Bool=false) where K
    if disk_cache && VERSION < v"1.12-"
        error("disk_cache=true requires Julia 1.12+ (object_build_id not available)")
    end
    CompilerCache{K}(tag, method_table,
                     Dict{Tuple{Core.CodeInstance, K}, Any}(),
                     ReentrantLock(),
                     disk_cache)
end
CompilerCache(tag::Symbol, method_table=nothing; disk_cache::Bool=false) =
    CompilerCache{Nothing}(tag, method_table; disk_cache)

"""
    cache_owner(cache::CompilerCache, keys=nothing) -> CacheOwner

Get the CacheOwner for a cache. Use this as your interpreter's cache token
with `CC.cache_owner(interp)`.
"""
cache_owner(cache::CompilerCache{K}, keys::K=nothing) where K = CacheOwner{K}(cache.tag, keys)

#==============================================================================#
# Method registration
#==============================================================================#

"""
    add_method(cache, f, arg_types, source) -> Method

Register a method with custom source IR in the cache's method table.

# Arguments
- `cache::CompilerCache` - The compiler cache instance
- `f::Function` - The function to add a method to
- `arg_types::Tuple` - Argument types for this method
- `source` - Custom IR to store (any type)

# Returns
The created `Method` object.
"""
function add_method(cache::CompilerCache, f::Function, arg_types::Tuple, source)
    mt = cache.method_table
    sig = Tuple{typeof(f), arg_types...}

    m = ccall(:jl_new_method_uninit, Any, (Any,), parentmodule(f))

    m.name = nameof(f)
    m.module = parentmodule(f)
    m.file = Symbol("foreign")
    m.line = Int32(0)
    m.sig = sig
    m.nargs = Int32(1 + length(arg_types))
    m.isva = false
    m.called = UInt32(0)
    m.nospecialize = UInt32(0)
    m.external_mt = mt
    m.slot_syms = ""
    m.source = source

    ccall(:jl_method_table_insert, Cvoid, (Any, Any, Any), mt, m, nothing)

    return m
end

#==============================================================================#
# Method lookup
#==============================================================================#

# Before JuliaLang/julia#60718, `jl_method_lookup_by_tt` did not correctly cache overlay
# methods, causing lookups to fail or return stale global entries, so don't use the cache.
@static if false
    using Base: method_instance
else
    function method_instance(@nospecialize(f), @nospecialize(tt);
                             world::UInt=Base.get_world_counter(),
                             method_table::Union{Core.MethodTable,Nothing}=nothing)
        sig = Base.signature_type(f, tt)
        match, _ = CC._findsup(sig, method_table, world)
        match === nothing && return nothing
        CC.specialize_method(match)::Core.MethodInstance
    end
end

"""
    method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the MethodInstance for function `f` with argument types `tt`.

Uses Julia's cached method lookup (`jl_method_lookup_by_tt`) for fast lookups.
Returns `nothing` if no matching method is found.
"""
method_instance

#==============================================================================#
# Inference helpers
#==============================================================================#

"""
    populate!(cache, interp, mi) -> Vector{Pair{CodeInstance, Union{CodeInfo, Nothing}}}

Populate the code cache with CodeInstances for `mi` and its callees.

Runs type inference on `mi` using the provided interpreter, which must implement
the `CC.AbstractInterpreter` interface. The resulting CodeInstances are stored
in the cache for later retrieval.

Returns a vector of (CodeInstance, IR) pairs where:
- Native 1.12+: `[ci => CodeInfo, ...]` for root + callees
- Native 1.11: `[ci => nothing]` (codegen uses callback-based path)

The root CI is always the first entry: `first(result)[1]`

# Arguments
- `cache::CompilerCache` - The compiler cache instance to populate
- `interp` - Your AbstractInterpreter implementation
- `mi` - The MethodInstance to infer

# Example
```julia
struct MyInterpreter <: CC.AbstractInterpreter
    # ...
end

mi = method_instance(f, (Int,); world, method_table=cache.method_table)
interp = MyInterpreter(...)
codeinfos = populate!(cache, interp, mi)
ci = first(codeinfos)[1]  # Root CodeInstance
```
"""
function populate!(cache::CompilerCache, interp::CC.AbstractInterpreter,
                   mi::Core.MethodInstance)
    @static if VERSION >= v"1.12.0-DEV.1434"
        # Modern API: returns CodeInstance, use SOURCE_MODE to control caching
        # (SOURCE_MODE_FORCE_SOURCE was renamed to SOURCE_MODE_GET_SOURCE)
        source_mode = @static if isdefined(CC, :SOURCE_MODE_GET_SOURCE)
            CC.SOURCE_MODE_GET_SOURCE
        else
            CC.SOURCE_MODE_FORCE_SOURCE
        end
        ci = CC.typeinf_ext(interp, mi, source_mode)
        @assert ci !== nothing "Inference of $mi failed"

        # Collect all code that needs compilation (including callees)
        codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]

        @static if VERSION >= v"1.13.0-DEV.499" || v"1.12-beta3" <= VERSION < v"1.13-"
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, ci)
            while !isempty(workqueue)
                callee = pop!(workqueue)
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    @static if VERSION >= v"1.13.0-DEV.1121"
                        src = CC.codeinfo_for_const(interp, callee_mi,
                            CC.WorldRange(callee.min_world, callee.max_world),
                            callee.edges, callee.rettype_const)
                    else
                        src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                    end
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    sptypes = CC.sptypes_from_meth_instance(callee_mi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                    push!(codeinfos, callee => src)
                end
            end
        else
            # Older 1.12 API
            workqueue = Core.CodeInstance[ci]
            inspected = IdSet{Core.CodeInstance}()
            while !isempty(workqueue)
                callee = pop!(workqueue)
                callee in inspected && continue
                push!(inspected, callee)

                callee_mi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    src = CC.codeinfo_for_const(interp, callee_mi, callee.rettype_const)
                else
                    src = CC.typeinf_code(interp, callee_mi, true)
                end
                if src isa Core.CodeInfo
                    CC.collectinvokes!(workqueue, src)
                    push!(codeinfos, callee => src)
                end
            end
        end

        return codeinfos
    elseif VERSION >= v"1.12.0-DEV.15"
        # Julia 1.12 early API
        source_mode = @static if isdefined(CC, :SOURCE_MODE_GET_SOURCE)
            CC.SOURCE_MODE_GET_SOURCE
        else
            CC.SOURCE_MODE_FORCE_SOURCE
        end
        ci = CC.typeinf_ext_toplevel(interp, mi, source_mode)
        @assert ci !== nothing "Inference of $mi failed"
        return Pair{Core.CodeInstance, Core.CodeInfo}[]
    else
        # Julia 1.11 API - cache populated implicitly
        src = CC.typeinf_ext_toplevel(interp, mi)

        # Handle const-return case where ci.inferred may be nothing
        world = @static if isdefined(CC, :get_inference_world)
            CC.get_inference_world(interp)
        else
            CC.get_world_counter(interp)
        end
        ci = lookup(cache, mi; world)
        @assert ci !== nothing "Inference of $mi failed"
        if ci.inferred === nothing
            @atomic ci.inferred = src
        end

        # Return consistent type: [ci => nothing] (codegen uses callback path)
        return Pair{Core.CodeInstance, Union{Core.CodeInfo, Nothing}}[ci => nothing]
    end
end

#==============================================================================#
# Internal cache helpers
#==============================================================================#

@static if VERSION >= v"1.14-"
    function code_cache(owner::CacheOwner, world::UInt)
        world_range = CC.WorldRange(world)
        return CC.InternalCodeCache(owner, world_range)
    end
else
    function code_cache(owner::CacheOwner, world::UInt)
        cache = CC.InternalCodeCache(owner)
        return CC.WorldView(cache, world)
    end
end

function lookup(cache::CompilerCache{K}, mi::Core.MethodInstance, keys::K=nothing;
                world::UInt=Base.get_world_counter()) where K
    owner = cache_owner(cache, keys)
    cc = code_cache(owner, world)
    return CC.get(cc, mi, nothing)
end

"""
    store_backedges(mi::MethodInstance, ci::CodeInstance, deps::Vector{MethodInstance})

Register backedges so Julia automatically invalidates cached code when dependencies change.
This enables Julia's built-in invalidation mechanism - when any dependency MI is
invalidated, the caller MI's CodeInstances will have their max_world reduced.

Note: The API changed between Julia versions:
- Julia 1.11: jl_method_instance_add_backedge takes MethodInstance as caller
- Julia 1.12+: jl_method_instance_add_backedge takes CodeInstance as caller
"""
function store_backedges(mi::Core.MethodInstance, ci::Core.CodeInstance,
                         deps::Vector{Core.MethodInstance})
    isa(mi.def, Method) || return  # don't add backedges to toplevel

    for dep_mi in deps
        @static if VERSION >= v"1.12-"
            # Julia 1.12+: pass CodeInstance as caller
            ccall(:jl_method_instance_add_backedge, Cvoid,
                  (Any, Any, Any), dep_mi, nothing, ci)
        else
            # Julia 1.11: pass MethodInstance as caller
            ccall(:jl_method_instance_add_backedge, Cvoid,
                  (Any, Any, Any), dep_mi, nothing, mi)
        end
    end
    nothing
end

"""
    cache!(cache, mi, keys=nothing; world, deps) -> CodeInstance

Create and store a CodeInstance for `mi` in the cache.

Used for foreign mode where inference doesn't run. The CI participates in
Julia's invalidation mechanism via backedges registered from `deps`.

# Arguments
- `cache::CompilerCache{K}` - The compiler cache instance
- `mi::MethodInstance` - The method instance to cache
- `keys::K` - Sharding keys (default: nothing)
- `world::UInt` - World age for the CI
- `deps::Vector{MethodInstance}` - Dependencies to register as backedges
"""
function cache!(cache::CompilerCache{K}, mi::Core.MethodInstance, keys::K=nothing;
                world::UInt=Base.get_world_counter(),
                deps::Vector{Core.MethodInstance}=Core.MethodInstance[]) where K
    owner = cache_owner(cache, keys)
    cc = code_cache(owner, world)
    edges = isempty(deps) ? Core.svec() : Core.svec(deps...)

    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), nothing, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, nothing,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), UInt32(0), nothing, UInt8(0))
    end
    CC.setindex!(cc, ci, mi)

    # Register backedges for automatic invalidation
    if !isempty(deps)
        store_backedges(mi, ci, deps)
    end

    return ci
end

#==============================================================================#
# Disk cache I/O helpers
#==============================================================================#

"""
    disk_cache_path() -> String

Return the path to the disk cache directory.
"""
disk_cache_path() = @get_scratch!("disk_cache")

"""
    clear_disk_cache!(cache::CompilerCache)

Remove disk-cached compilation results for the given cache (by tag).
"""
clear_disk_cache!(cache::CompilerCache) =
    rm(joinpath(disk_cache_path(), string(cache.tag)); recursive=true, force=true)

"""
    cache_file(cache, ci, keys) -> Union{String, Nothing}

Return the path to the cache file for the given CodeInstance and keys.

On Julia 1.12+, uses `Base.object_build_id(ci)` as the cache key:
- Stable across sessions for precompiled package code
- Different after method redefinition (new CI = new build_id)
- Returns `nothing` for runtime compilations (build_id === nothing)

On Julia 1.11, returns `nothing` to skip disk caching. Without `object_build_id`,
we cannot safely distinguish between different versions of the same method,
which would cause stale bitcode to be loaded after method redefinition.
"""
function cache_file(cache::CompilerCache{K}, ci::Core.CodeInstance,
                    keys::K) where K
    @static if VERSION >= v"1.12-"
        # Use the MethodInstance's build_id, not the CodeInstance's.
        # CodeInstances created at runtime (via populate!) have build_id === nothing,
        # but their MethodInstance retains the stable build_id from precompilation.
        mi = ci.def
        bid = Base.object_build_id(mi)
        bid === nothing && return nothing  # runtime MI, skip disk cache

        # Use only stable identifiers for disk cache key:
        # - object_build_id(mi): stable across sessions for precompiled methods
        # - specTypes: identifies the specific method instance (redundant but safe)
        # - keys: sharding keys
        # NOTE: Do NOT use objectid() - it's a memory address that changes between sessions
        h = hash(bid)
        h = hash(mi.specTypes, h)
        h = hash(keys, h)
        return joinpath(disk_cache_path(), string(cache.tag), string(h, ".jls"))
    else
        # Julia 1.11: no object_build_id, skip disk caching
        # Using specTypes would load stale bitcode after method redefinition
        return nothing
    end
end

"""
    write_disk_cache(path, mi, keys, data)

Atomically write a cache entry to disk.
"""
function write_disk_cache(path::String, mi::Core.MethodInstance,
                          keys::K, data) where K
    mkpath(dirname(path))
    entry = DiskCacheEntry{K}(mi.specTypes, keys, data)
    tmppath, io = mktemp(dirname(path); cleanup=false)
    try
        serialize(io, entry)
        close(io)
        mv(tmppath, path; force=true)
    catch
        close(io)
        rm(tmppath; force=true)
        rethrow()
    end
end

"""
    read_disk_cache(path, mi, keys) -> Union{Any, Nothing}

Read a cache entry from disk with validation.
Returns `nothing` if validation fails.
"""
function read_disk_cache(path::String, mi::Core.MethodInstance,
                         keys::K) where K
    isfile(path) || return nothing
    try
        entry = deserialize(path)::DiskCacheEntry
        if entry.spec_types == mi.specTypes && entry.keys == keys
            return entry.data
        end
        @warn "Disk cache mismatch" mi path
        return nothing
    catch e
        @warn "Failed to read disk cache" path exception=e
        return nothing
    end
end

#==============================================================================#
# Cached compilation - main API
#==============================================================================#

"""
    cached_compilation(cache, mi, world, [keys]; infer, codegen, link) -> result

Three-phase cached compilation with automatic invalidation and optional disk persistence.

# Three-Phase Design

The compilation pipeline is split into three phases to enable caching at different granularities:

## Phase 1: `infer(cache, mi, world) -> Vector{Pair{CodeInstance, Any}}`

**Purpose**: Establish the compilation unit and its dependencies.

**Returns**: A vector of `CodeInstance => IR` pairs where:
- The first entry's CodeInstance becomes the cache key
- The IR can be any type (CodeInfo for native, custom IR for foreign compilers)

**Why separate**: The CodeInstance provides:
- A stable identity for cache lookup (`object_build_id` for disk caching)
- Automatic invalidation tracking via Julia's backedge mechanism
- World age bounds (`min_world`/`max_world`) for validity checking

**Native mode**: Call `populate!(cache, interp, mi)` to run Julia's type inference.
The returned CodeInstances have edges tracking all called methods.

**Foreign mode**: Call `cache!(cache, mi; world, deps)` to create a CodeInstance
with explicit dependency tracking. Pass the MethodInstances of any callees as `deps`.

## Phase 2: `codegen(cache, mi, world, codeinfos) -> ir_data`

**Purpose**: Generate serializable intermediate representation.

**Input**: The `codeinfos` vector from the infer phase.

**Returns**: Serializable IR data (e.g., LLVM bitcode, custom bytecode).

**Why separate**: This is typically the expensive phase. Separating it enables:
- Disk caching of the IR (keyed by CodeInstance's `object_build_id`)
- Skipping codegen entirely on disk cache hit
- Cross-session persistence of compilation work

**Requirements**: The returned data must be serializable if `disk_cache=true`.

## Phase 3: `link(cache, mi, world, ir_data) -> result`

**Purpose**: Create the final session-specific result.

**Input**: The `ir_data` from codegen (or disk cache).

**Returns**: The final compilation result (e.g., function pointer, executable handle).

**Why separate**: Link results are inherently session-specific:
- Function pointers are valid only in the current process
- JIT'd machine code lives in process memory
- Must be re-executed each session, even with disk cache hits

The link result is stored in memory cache for reuse within the session.

# Cache Behavior

1. **Memory cache hit**: Return immediately (no phases run)
2. **Memory miss, disk hit**: Run only `link` phase
3. **Full miss**: Run all three phases

# Arguments

- `cache::CompilerCache{K}` - Cache instance (set `disk_cache=true` for persistence)
- `mi::MethodInstance` - The method instance to compile
- `world::UInt` - World age for cache lookup and validity
- `keys::K` - Optional sharding keys (default: nothing)

# Keyword Arguments

- `infer` - Inference function: `(cache, mi, world) -> Vector{Pair{CodeInstance, Any}}`
- `codegen` - Code generation function: `(cache, mi, world, codeinfos) -> ir_data`
- `link` - Linking function: `(cache, mi, world, ir_data) -> result`

# Example

```julia
# Native compilation with Julia's inference
cached_compilation(cache, mi, world;
    infer = (c, m, w) -> populate!(c, MyInterpreter(c, w), m),
    codegen = (c, m, w, cis) -> emit_llvm_ir(cis),
    link = (c, m, w, ir) -> jit_compile(ir)
)

# Foreign IR compilation with explicit dependencies
cached_compilation(cache, mi, world;
    infer = (c, m, w) -> begin
        ir = m.def.source::MyIR
        deps = [method_instance(f, tt; world=w) for (f, tt) in ir.calls]
        ci = cache!(c, m; world=w, deps)
        [ci => ir]
    end,
    codegen = (c, m, w, cis) -> compile_my_ir(only(cis)[2]),
    link = (c, m, w, result) -> result
)
```
"""
function cached_compilation(cache::CompilerCache{K}, mi::Core.MethodInstance,
                            world::UInt, keys::K=nothing;
                            infer, codegen, link) where K
    result = nothing

    # 1. Check memory cache
    ci = lookup(cache, mi, keys; world)
    if ci !== nothing
        key = (ci, keys)
        lock(cache.lock) do
            result = get(cache.external_cache, key, nothing)
        end
    end
    result !== nothing && return result

    # 2. Run inference to get codeinfos (root CI is first entry)
    #    infer() calls populate! or cache! depending on mode
    codeinfos = nothing
    if ci === nothing
        codeinfos = infer(cache, mi, world)
        ci, _ = first(codeinfos)
        @assert ci !== nothing "Inference failed to produce a CodeInstance"
    end

    # 3. Check disk cache using CI's build_id
    ir_data = nothing
    if cache.disk_cache
        path = cache_file(cache, ci, keys)
        if path !== nothing
            ir_data = read_disk_cache(path, ci.def, keys)
        end
    end

    # 4. If disk miss, run codegen
    if ir_data === nothing
        # Need codeinfos for codegen - if we had a memory cache hit on CI but
        # disk miss, we need to re-run inference to get codeinfos
        if codeinfos === nothing
            codeinfos = infer(cache, mi, world)
        end

        ir_data = codegen(cache, mi, world, codeinfos)

        # Write to disk cache if enabled
        if cache.disk_cache
            path = cache_file(cache, ci, keys)
            if path !== nothing
                write_disk_cache(path, ci.def, keys, ir_data)
            end
        end
    end

    # 5. Link and store in memory cache
    result = link(cache, mi, world, ir_data)
    key = (ci, keys)
    lock(cache.lock) do
        cache.external_cache[key] = result
    end

    return result
end

end # module CompilerCaching
