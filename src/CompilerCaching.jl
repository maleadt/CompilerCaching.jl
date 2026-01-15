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
export add_method, cached_compilation, methodinstance, register_dependency!

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

#==============================================================================#
# CompilationContext - passed to compiler
#==============================================================================#

"""
    CompilationContext

Context passed to the compile function on cache miss.

- `mi::Core.MethodInstance` - The method instance being compiled
- Use `register_dependency!(ctx, other_mi)` to track dependencies
"""
mutable struct CompilationContext
    mi::Core.MethodInstance
    deps::Vector{Core.MethodInstance}
end

CompilationContext(mi) = CompilationContext(mi, Core.MethodInstance[])

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
    methodinstance(cache, f, tt; world) -> Union{MethodInstance, Nothing}

Look up a method instance for function `f` with argument types `tt`.

Uses `cache.mt` for lookup if set, otherwise uses the global method table.

# Returns
- `MethodInstance` if a matching method is found
- `nothing` if no matching method exists
"""
function methodinstance(cache::CompilerCache, f::Function, @nospecialize(tt);
                       world::Integer=Base.get_world_counter())
    ft = Core.Typeof(f)
    sig = Tuple{ft, tt...}

    if cache.mt === nothing
        # Global MT lookup
        results = Base._methods_by_ftype(sig, -1, world)
    else
        # Custom MT lookup
        results = Base._methods_by_ftype(sig, cache.mt, -1, world)
    end

    isempty(results) && return nothing
    m = results[1].method

    return CC.specialize_method(m, sig, Core.svec())::Core.MethodInstance
end

#==============================================================================#
# Cache access
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

#==============================================================================#
# Cached compilation - main API
#==============================================================================#

"""
    cached_compilation(cache::CompilerCache{Nothing}, f, tt, compiler) -> Union{Some, Nothing}

Look up or compile code for function `f` with argument types `tt`.
For CompilerCache with no sharding keys.

# Arguments
- `cache` - CompilerCache instance
- `f` - Function to compile
- `tt` - Argument types tuple
- `compiler(ctx) -> result` - Called on cache miss. Access source via `ctx.mi.def.source`.

# Returns
- `Some(result)` - Compilation result wrapped in `Some` (use `something(result)` to unwrap)
- `nothing` - No matching method found for the given function and argument types
"""
function cached_compilation(compiler, cache::CompilerCache{Nothing}, f::Function,
                            @nospecialize(tt))
    owner = CacheOwner{Nothing}(cache.tag, nothing)
    _cached_compilation(cache, owner, f, tt, compiler)
end

"""
    cached_compilation(cache::CompilerCache{K}, f, tt, keys::K, compiler) -> Union{Some, Nothing}

Look up or compile code for function `f` with argument types `tt` and sharding keys.

# Arguments
- `cache` - CompilerCache instance with key type K
- `f` - Function to compile
- `tt` - Argument types tuple
- `keys` - Sharding keys (must match K type)
- `compiler(ctx) -> result` - Called on cache miss. Access source via `ctx.mi.def.source`.

# Returns
- `Some(result)` - Compilation result wrapped in `Some` (use `something(result)` to unwrap)
- `nothing` - No matching method found for the given function and argument types
"""
function cached_compilation(compiler, cache::CompilerCache{K}, f::Function,
                            @nospecialize(tt), keys::K) where K
    owner = CacheOwner{K}(cache.tag, keys)
    _cached_compilation(cache, owner, f, tt, compiler)
end

# Internal implementation
function _cached_compilation(cache::CompilerCache, owner::CacheOwner,
                            f::Function, @nospecialize(tt), compiler)
    world = Base.get_world_counter()
    mi = methodinstance(cache, f, tt; world)

    # No matching method found
    mi === nothing && return nothing

    # Fast path: check cache
    cc = code_cache(owner, world)
    ci = CC.get(cc, mi, nothing)
    if ci !== nothing
        return Some(ci.inferred)
    end

    # Slow path: compile
    ctx = CompilationContext(mi)
    result = compiler(ctx)

    # Create CodeInstance with dependencies as edges
    edges = isempty(ctx.deps) ? Core.svec() : Core.svec(ctx.deps...)
    @static if VERSION >= v"1.12-"
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, result,
            Int32(0), world, typemax(UInt), UInt32(0), nothing, nothing, edges)
    else
        ci = Core.CodeInstance(mi, owner, Any, Any, nothing, result,
            Int32(0), world, typemax(UInt), UInt32(0), UInt32(0), nothing, UInt8(0))
    end
    CC.setindex!(cc, ci, mi)

    return Some(result)
end

end # module CompilerCaching
