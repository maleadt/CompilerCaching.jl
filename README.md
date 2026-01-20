# CompilerCaching.jl

A package for interfacing with Julia's compiler caching infrastructure for the purpose
of building custom compilers.


## Installation

```julia
using Pkg
Pkg.add(url="path/to/CompilerCaching")
```


## Usage

The basic usage pattern is to create a `CacheHandle` handle and invoke `cached_compilation`
with three callbacks:
- `infer`: Perform type inference, store a `CodeInstance` in the cache (via `populate!` or `cache!`), and return the data for codegen
- `codegen`: Generate serializable code that can be cached across sessions
- `link`: Generate a session-specific representation (e.g., JIT-compiled function pointer)

A `CacheHandle` is a lightweight handle to Julia's global `InternalCodeCache`. Creating
cache handles is cheap, so they can be constructed on-the-fly right before compilation.

```julia
using CompilerCaching

# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheHandle
    world::UInt
    ...
end
@setup_caching CustomInterpreter.cache

function infer(cache, mi, world)
    # Let Julia populate the cache and return codeinfos for codegen
    interp = CustomInterpreter(cache, world)
    return CompilerCaching.populate!(cache, interp, mi)
end

# generate some IR representation
function codegen(cache, mi, world, codeinfos) end

# compile IR to function pointer
function link(cache, mi, world, result) end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    cached_compilation(cache, mi, world; infer, codegen, link)
end

my_function(x::Int) = x + 100
call(my_function, 42)
```

The `@setup_caching` macro defines the necessary methods to connect the interpreter
to the cache, and can be customized further if needed.

### Cache sharding

It is possible to partition the cache by additional parameters by storing `keys` in the
cache handle, e.g., encoding the optimization level or target device:

```julia
const CacheKey = @NamedTuple{opt_level::Int}

function call(f, args...; opt_level=1)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheHandle{CacheKey}(:MyCompiler, (; opt_level))
    cached_compilation(cache, mi, world; infer, codegen, link)
end
```

Different calls with the same `(tag, keys)` will hit the same cache partition.

### Overlay methods

It is often useful to redefine existing methods for use with the custom compiler.
This can be accomplished using overlay methods in a custom method table:

```julia
Base.Experimental.@MethodTable method_table
Base.Experimental.@overlay method_table function Base.sin(x::Int)
    # custom implementation
end

# Expose the method table to the interpreter
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheHandle
    world::UInt
    ...
end
CC.method_table(interp::CustomInterpreter) = CC.OverlayMethodTable(interp.world, method_table)

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    # if relevant, look for global methods too
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    cached_compilation(cache, mi, world; infer, codegen, link)
end
```

### Foreign IR

For compilers that define their own IR format that Julia doesn't know about:

```julia
Base.Experimental.@MethodTable method_table

# Add a method to the method table, providing a custom IR source
function really_special end
add_method(method_table, really_special, (Int,), MyCustomIR([:a, :b]))

# Since we're not relying on Julia's inference, we need to create and cache our own CIs
function infer(cache, mi, world)
    ir = mi.def.source::MyCustomIR
    deps = Core.MethodInstance[]
    for callee in ir.callees
        callee_mi = method_instance(callee.f, callee.tt; world, method_table)
        cached_inference(cache, callee_mi, world; infer)
        push!(deps, callee_mi)
    end
    cache!(cache, mi; world, deps)
    return ir
end

function codegen(cache, mi, world, codeinfos) end

function link(cache, mi, world, result) end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    cached_compilation(cache, mi, world; infer, codegen, link)
end
```

### Stacked method tables

A utility `StackedMethodTable` is provided to facilitate layering multiple method tables:

```julia

MyMethodTableStack(world) = StackedMethodTable(world, overlay_table, base_table)

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
end
CC.method_table(interp::CustomInterpreter) = MyMethodTableStack(interp.world)
```
