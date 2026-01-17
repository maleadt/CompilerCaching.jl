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
const CC = Core.Compiler

include("utils.jl")

export CompilerCache, CacheOwner, CompilationContext
export add_method, cached_compilation, method_instance, register_dependency!
export cache_owner, cache!
export populate!, StackedMethodTable

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
end

# Constructors
function CompilerCache{K}(tag::Symbol, method_table) where K
    CompilerCache{K}(tag, method_table,
                     Dict{Tuple{Core.CodeInstance, K}, Any}(),
                     ReentrantLock())
end
CompilerCache(tag::Symbol, method_table) = CompilerCache{Nothing}(tag, method_table)
CompilerCache(tag::Symbol) = CompilerCache{Nothing}(tag, nothing)
CompilerCache{K}(tag::Symbol) where K = CompilerCache{K}(tag, nothing)  # Global MT with sharding keys

"""
    cache_owner(cache::CompilerCache, keys=nothing) -> CacheOwner

Get the CacheOwner for a cache. Use this as your interpreter's cache token
with `CC.cache_owner(interp)`.
"""
cache_owner(cache::CompilerCache{K}, keys::K=nothing) where K = CacheOwner{K}(cache.tag, keys)

#==============================================================================#
# CompilationContext - passed to compiler
#==============================================================================#

"""
    CompilationContext

Context passed to the compile function on cache miss.

- Use `register_dependency!(ctx, other_mi)` to track dependencies
"""
mutable struct CompilationContext
    deps::Vector{Core.MethodInstance}

    CompilationContext() = new(Core.MethodInstance[])
end

"""
    register_dependency!(ctx, mi)

Register that the current compilation depends on `mi`.
When `mi` is invalidated, this compilation will also be invalidated.
"""
function register_dependency!(ctx::CompilationContext, mi::Core.MethodInstance)
    push!(ctx.deps, mi)
end

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
    populate!(cache, interp, mi) -> Union{Vector{Pair{CodeInstance, CodeInfo}}, Nothing}

Populate the code cache with CodeInstances for `mi` and its callees.

Runs type inference on `mi` using the provided interpreter, which must implement
the `CC.AbstractInterpreter` interface. The resulting CodeInstances are stored
in the cache for later retrieval.

On Julia 1.12+, returns a vector of (CodeInstance, CodeInfo) pairs for codegen.
On Julia 1.11, returns `nothing` (cache is populated implicitly by typeinf_ext_toplevel).

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
result = populate!(cache, interp, mi)
# result is Vector{Pair{CodeInstance, CodeInfo}} on 1.12+, nothing on 1.11
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
        # Julia 1.11 API - cache populated implicitly, return nothing
        src = CC.typeinf_ext_toplevel(interp, mi)

        # Handle const-return case where ci.inferred may be nothing
        world = @static if isdefined(CC, :get_inference_world)
            CC.get_inference_world(interp)
        else
            CC.get_world_counter(interp)
        end
        ci = lookup(cache, mi; world)
        if ci !== nothing && ci.inferred === nothing
            @atomic ci.inferred = src
        end

        return nothing
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
# Cached compilation - main API
#==============================================================================#

"""
    cached_compilation(compiler, cache, mi, world, keys=nothing) -> result

Look up or compile code for MethodInstance `mi`.

# Arguments
- `compiler(ctx) -> result` - Called on cache miss. Use `register_dependency!(ctx, mi)` to track dependencies.
- `cache::CompilerCache{K}` - Cache instance
- `mi::MethodInstance` - The method instance to compile
- `world::UInt` - World age for cache lookup
- `keys::K` - Sharding keys (default: nothing)

# Returns
The compilation result (from cache or freshly compiled).

# Example
```julia
world = Base.get_world_counter()
mi = method_instance(f, tt; world, method_table=cache.method_table)
mi === nothing && throw(MethodError(f, tt))

result = cached_compilation(cache, mi, world) do ctx
    compile(mi)  # mi passed directly, not ctx.mi
end
```
"""
function cached_compilation(compiler, cache::CompilerCache{K},
                            mi::Core.MethodInstance, world::UInt,
                            keys::K=nothing) where K
    result = nothing

    # Fast path: find CI and check cache
    # Note: lookup() uses Julia's cache which checks max_world,
    # so invalidated CIs will return nothing automatically
    ci = lookup(cache, mi, keys; world)
    if ci !== nothing
        key = (ci, keys)
        lock(cache.lock) do
            result = get(cache.external_cache, key, nothing)
        end
    end

    # Slow path: compile
    if result === nothing
        ctx = CompilationContext()
        result = compiler(ctx)

        # Look up CI (inference may have created it)
        if ci === nothing
            ci = lookup(cache, mi, keys; world)
        end

        # For foreign mode (no inference), create CI ourselves with backedges
        if ci === nothing
            ci = cache!(cache, mi, keys; world, deps=ctx.deps)
        end
        key = (ci, keys)

        lock(cache.lock) do
            cache.external_cache[key] = result
        end
    end

    return result
end

end # module CompilerCaching
