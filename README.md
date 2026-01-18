# CompilerCaching.jl

A package for interfacing with Julia's compiler caching infrastructure for the purpose
of building custom compilers.


## Features

- **Lazy compilation with caching**: Only compile when needed, cache results for reuse
- **Type-based specialization**: Automatic dispatch based on argument types
- **Automatic invalidation**: Cache entries invalidated when methods are redefined
- **Transitive dependency tracking**: Register dependencies to propagate invalidation
- **Sharding keys**: Partition caches by additional parameters (optimization level, target device, etc.)

Requires Julia 1.11+.

## Installation

```julia
using Pkg
Pkg.add(url="path/to/CompilerCaching")
```

## Examples

Basic example:

```julia
using CompilerCaching

my_function(x::Int) = x + 100

cache = CompilerCache(:MyCompiler)

function infer(cache, mi, world)
    # Perform type inference, constructing a code instance with invalidation edges
    ci = cache!(cache, mi; world)
    [ci => :inferred]
end

function codegen(cache, mi, world, codeinfos)
    # Generate serializable code
    mi.def.n
end

function link(cache, mi, world, result)
    # Comple to a session-specific representation
    result
end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, cache.method_table),
                    throw(MethodError(f, args)))

    cached_compilation(cache, mi, world; infer, codegen, link)
end
call(my_function, 42)
```

### Cache sharding

Partition caches by additional parameters (e.g., optimization level, target device):

```julia
using CompilerCaching

my_function(x::Int) = x + 100

const CacheKey = @NamedTuple{opt_level::Int}
cache = CompilerCache{CacheKey}(:MyCompiler)

function call(f, args...; opt_level=1)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world).
                    throw(MethodError(f, args)))

    cached_compilation(cache, mi, world, (opt_level=opt_level,);
                       infer, codegen, link)
end
```


### Overlay methods

This can be useful when working with methods that are not available in the global method table:

```julia
using CompilerCaching
using Base.Experimental: @MethodTable

@MethodTable method_table

function device_kernel end
Base.Experimental.@overlay method_table function device_kernel(x::Int)
    x * 2
end
device_kernel(args...) = error("Not for direct use.")

cache = CompilerCache(:MyCompiler, method_table)

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi @something(method_instance(f, tt; world, cache.method_table)
                  throw(MethodError(f, args)))

    cached_compilation(cache, mi, world; infer, codegen, link)
end
```

### Foreign IR with dependency tracking

For compilers that define their own IR format that Julia doesn't know about:

```julia
using CompilerCaching
using Base.Experimental: @MethodTable

@MethodTable method_table
cache = CompilerCache(:MyCompiler, method_table)

function really_special end
add_method(cache, really_special, (Int,), MyCustomIR([:a, :b]))

# Infer phase: handles dependency tracking via cache!
function foreign_infer(cache, mi, world)
    ir = mi.def.source::MyCustomIR
    deps = Core.MethodInstance[]

    # Recursively compile any callees and collect deps
    for callee in ir.callees
        callee_mi = method_instance(callee.f, callee.tt;
                                    world, method_table=cache.method_table)
        cached_compilation(cache, callee_mi, world;
            infer = foreign_infer,
            codegen = foreign_codegen,
            link = foreign_link)
        push!(deps, callee_mi)
    end

    # Create CI with backedges for dependency tracking
    ci = cache!(cache, mi; world, deps)
    return [ci => ir]
end

function foreign_codegen(cache, mi, world, codeinfos)
    _, ir = only(codeinfos)
    compile_my_ir(ir)
end

function foreign_link(cache, mi, world, result)
    result
end
```

Dependency tracking is handled by the `infer` phase via `cache!`, which registers backedges for automatic invalidation.


## API Reference

### Types

- `CompilerCache{K}` - Main cache instance parameterized by key type K
- `CacheOwner{K}` - Cache partition identifier (internal)

### Functions

- `CompilerCache(tag::Symbol, mt)` - Create cache with custom method table
- `CompilerCache(tag::Symbol)` - Create cache using global method table
- `CompilerCache{K}(tag::Symbol)` - Create cache with sharding keys and global MT
- `add_method(cache, f, arg_types, source)` - Register method with custom source
- `cached_compilation(cache, mi, world, [keys]; infer, codegen, link)` - Three-phase compilation API
- `method_instance(f, tt; world, method_table)` - Get MethodInstance for function + arg types
- `cache!(cache, mi; world, deps)` - Create CodeInstance with dependency tracking

## How It Works

CompilerCaching leverages Julia's internal compiler infrastructure:

1. **Method Tables**: Custom `Core.MethodTable` for isolated method dispatch
2. **MethodInstance**: Specialized method for specific argument types
3. **CodeInstance**: Cached compilation result with validity tracking
4. **World Age**: Automatic invalidation when methods are redefined
5. **Backedges**: Transitive invalidation via dependency tracking

The package uses `Core.Compiler.InternalCodeCache` for storage, which integrates with Julia's garbage collection and invalidation mechanisms.
