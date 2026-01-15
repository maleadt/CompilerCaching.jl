# CompilerCaching.jl

A reusable package for caching compilation results using Julia's `Method`/`MethodInstance`/`CodeInstance` infrastructure.

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

## Eamples

Basic example:

```julia
using CompilerCaching

my_function(x::Int) = x + 100

cache = CompilerCache(:MyCompiler)

function call(f, args...)
    tt = map(Core.Typeof, args)
    result = cached_compilation(cache, f, tt) do ctx
        run_inference_and_codegen(ctx.mi)
    end
    compiled = @something result throw(MethodError(f, args))

    compiled(args...)
end
call(my_function, 42)
```

<!-- TODO: this shouldn't call a magical run_inference_and_codegen but actually use Julia codegen -->

<!-- how does this work for nested function calls? Doesn't Julia normally do nested copmpilation automatically? how does transient invalidation work? -->

### Cache sharding

In case you want to compile a single method instance differently, e.g., by doing different optimization levels, you can shard the cache on arbitrary (immutable) keys:

```julia
using CompilerCaching

my_function(x::Int) = x + 100

const CacheKey = @NamedTuple{opt_level::Int}
cache = CompilerCache{CacheKey}(:MyCompiler)

function call(f, args...)
    tt = map(Core.Typeof, args)
    params = CacheKey(1)
    result = cached_compilation(cache, f, tt, params) do ctx
        run_inference_and_codegen(ctx.mi)
    end
    compiled = @something result throw(MethodError(f, args))

    compiled(args...)
end
call(my_function, 42)
```

<!-- TODO: ensure immutability -->


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
    result = cached_compilation(cache, f, tt) do ctx
        run_inference_and_codegen(ctx.mi)
    end
    compiled = @something result throw(MethodError(f, args))

    compiled(args...)
end
call(my_function, 42)
```

### Foreign IR with custom dependency tracking

For compilers that define their own IR format that Julia doesn't know about:

```julia
using CompilerCaching
using Base.Experimental: @MethodTable

@MethodTable method_table

function really_special end
add_method(cache, really_special, (Int,), MyCustomIR([:a, :b]))

cache = CompilerCache(:MyCompiler, method_table)

function call(f, args...)
    tt = map(Core.Typeof, args)
    result = cached_compilation(cache, f, tt) do ctx
        ir = ctx.mi.def.source # custom IR
        # mark dependencies
        register_dependency!(ctx, called_mi)
        compile_my_ir(ir)
    end
    compiled = @something result throw(MethodError(f, args))

    compiled(args...)
end
call(really_special, 42)
```

Since inference isn't used, Julia doesn't know which functions are called, so we need to track these ourselves by caling `register_dependency!`.


## API Reference

### Types

- `CompilerCache{K}` - Main cache instance parameterized by key type K
- `CompilationContext` - Context passed to compile function
- `CacheOwner{K}` - Cache partition identifier (internal)

### Functions

- `CompilerCache(tag::Symbol, mt)` - Create cache with custom method table
- `CompilerCache(tag::Symbol)` - Create cache using global method table
- `CompilerCache{K}(tag::Symbol)` - Create cache with sharding keys and global MT
- `add_method(cache, f, arg_types, source)` - Register method with custom source
- `cached_compilation(cache, f, tt, [keys,] compile_fn)` - Look up or compile
- `method_instance(sig; world, method_table)` - Get MethodInstance by signature type
- `method_instance(f, tt; world, method_table)` - Get MethodInstance for function + arg types
- `register_dependency!(ctx, mi)` - Register transitive dependency

## How It Works

CompilerCaching leverages Julia's internal compiler infrastructure:

1. **Method Tables**: Custom `Core.MethodTable` for isolated method dispatch
2. **MethodInstance**: Specialized method for specific argument types
3. **CodeInstance**: Cached compilation result with validity tracking
4. **World Age**: Automatic invalidation when methods are redefined
5. **Backedges**: Transitive invalidation via dependency tracking

The package uses `Core.Compiler.InternalCodeCache` for storage, which integrates with Julia's garbage collection and invalidation mechanisms.
