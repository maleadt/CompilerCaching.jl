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

export CompilerCache, CacheOwner, CompilationContext
export add_method, cached_compilation, method_instance, register_dependency!
export cache_owner, cache!

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
- `mt::Union{Core.MethodTable, Nothing}` - Method table for dispatch (nothing = global MT)

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
    mt::Union{Core.MethodTable, Nothing}
end

# Constructors
CompilerCache(tag::Symbol, mt) = CompilerCache{Nothing}(tag, mt)
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
    mt = cache.mt
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

"""
    method_instance(f, tt; world, method_table) -> Union{MethodInstance, Nothing}

Look up the MethodInstance for function `f` with argument types `tt`.

Uses Julia's cached method lookup (`jl_method_lookup_by_tt`) for fast lookups.
Returns `nothing` if no matching method is found.
"""
function method_instance(@nospecialize(f), @nospecialize(tt);
                         world::UInt=Base.get_world_counter(),
                         method_table::Union{Core.MethodTable, Nothing}=nothing)
    sig = Base.signature_type(f, tt)
    mi = ccall(:jl_method_lookup_by_tt, Any,
               (Any, Csize_t, Any), sig, world, method_table)
    mi === nothing && return nothing
    return mi::Core.MethodInstance
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

function cache!(cache::CompilerCache{K}, mi::Core.MethodInstance, result, keys::K=nothing;
                world::UInt=Base.get_world_counter(), edges::Core.SimpleVector=Core.svec()) where K
    owner = cache_owner(cache, keys)
    cc = code_cache(owner, world)
    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, result,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), nothing, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, result,
            Int32(0), UInt(world), typemax(UInt), UInt32(0), UInt32(0), nothing, UInt8(0))
    end
    CC.setindex!(cc, ci, mi)
    return ci
end

#==============================================================================#
# Cached compilation - main API
#==============================================================================#

"""
    cached_compilation(compiler, cache, mi, world, keys=nothing) -> result

Look up or compile code for MethodInstance `mi`.

# Arguments
- `compiler(ctx) -> result` - Called on cache miss. Access source via `ctx.mi.def.source`.
- `cache::CompilerCache{K}` - Cache instance
- `mi::MethodInstance` - The method instance to compile
- `world::UInt` - World age for cache lookup
- `keys::K` - Sharding keys (default: nothing)

# Returns
The compilation result (from cache or freshly compiled).

# Example
```julia
world = Base.get_world_counter()
mi = method_instance(f, tt; world, method_table=cache.mt)
mi === nothing && throw(MethodError(f, tt))

result = cached_compilation(cache, mi, world) do ctx
    # compile ctx.mi
end
```
"""
function cached_compilation(compiler, cache::CompilerCache{K},
                            mi::Core.MethodInstance, world::UInt,
                            keys::K=nothing) where K
    # Fast path: cache hit
    ci = lookup(cache, mi, keys; world)
    if ci !== nothing
        return ci.inferred
    end

    # Slow path: compile and cache
    ctx = CompilationContext()
    result = compiler(ctx)
    edges = isempty(ctx.deps) ? Core.svec() : Core.svec(ctx.deps...)
    cache!(cache, mi, result, keys; world, edges)

    return result
end

end # module CompilerCaching
