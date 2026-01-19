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

## Usage

The very basic usage pattern is to create a `CompilerCache` instance, and invoke its
`cached_compilation` function passing along three callbacks to perform various aspects
of the compilation process:
- `infer`: Perform type inference and construct a `CodeInstance` with valid invalidation edges
- `codegen`: Generate serializable code that can be cached across sessions
- `link`: Generate a session-specific representation (e.g., JIT-compiled function pointer)

```julia
using CompilerCaching

const cache = CompilerCache(:MyCompiler)

# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    ...
end
CC.cache_owner(interp::CustomInterpreter) = cache_owner(interp.cache)

function infer(cache, mi, world)
    # Let Julia infer the method
    interp = MyInterpreter(cache, world)
    CompilerCaching.populate!(cache, interp, mi)
end

function codegen(cache, mi, world, codeinfos)
    # generate some IR representation
end

function link(cache, mi, world, result)
    # compile IR to function pointer
end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, cache.method_table),
                    throw(MethodError(f, args)))

    cached_compilation(cache, mi, world; infer, codegen, link)
end

my_function(x::Int) = x + 100
call(my_function, 42)
```

### Cache sharding

It is possible to partition the cache by additional parameters by passing `keys` to
`cached_compilation`, e.g., encoding the optimization level or target device:

```julia
const CacheKey = @NamedTuple{opt_level::Int}
const cache = CompilerCache{CacheKey}(:MyCompiler)

function call(f, args...; opt_level=1)
    # ...

    cached_compilation(cache, mi, world, (opt_level=opt_level,);
                       infer, codegen, link)
end
```

### Overlay methods

It is often useful to redefine existing methods for use with the custom compiler.
This can be accomplished using overlay methods in a custom method table:

```julia
Base.Experimental.@MethodTable method_table
Base.Experimental.@overlay method_table function Base.sin(x::Int)
    # custom implementation
end

const cache = CompilerCache(:MyCompiler, method_table)

# Expose the method table to the interpreter
struct CustomInterpreter <: CC.AbstractInterpreter
    ...
end
CC.method_table(interp::CustomInterpreter) = method_table

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()

    # if relevant, look for the entry point in the custom method table too
    mi = @something(method_instance(f, tt; world, cache.method_table),
                    method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cached_compilation(cache, mi, world; infer, codegen, link)
end
```


### Foreign IR

For compilers that define their own IR format that Julia doesn't know about:

```julia
Base.Experimental.@MethodTable method_table
const cache = CompilerCache(:MyCompiler, method_table)

# add a method to the method table and cache, providing a custom IR source
function really_special end
add_method(cache, really_special, (Int,), MyCustomIR([:a, :b]))

# Since we're not relying on Julia's inference, we need to create and cache our own CIs
function infer(cache, mi, world)
    ir = mi.def.source::MyCustomIR
    deps = Core.MethodInstance[]
    for callee in ir.callees
        callee_mi = method_instance(callee.f, callee.tt; world, cache.method_table)
        cached_compilation(cache, callee_mi, world; infer, codegen, link)
        push!(deps, callee_mi)
    end
    ci = cache!(cache, mi; world, deps)
    return [ci => ir]
end

function codegen(cache, mi, world, codeinfos)
    _, ir = only(codeinfos)
    compile_my_ir(ir)
end

function link(cache, mi, world, result)
    result
end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = method_instance(f, tt; world, method_table=FOREIGN_CACHE.method_table)
    mi === nothing && throw(MethodError(f, args))

    cached_compilation(FOREIGN_CACHE, mi, world; infer, codegen, link)
end
```


### Disk caching

On Julia 1.12 and up, it is possible to back the cache by a disk-based store:

```julia
const cache = CompilerCache(:MyCompiler; disk_cache=true)

# explicitly wipe the cache, if needed
clear_disk_cache!(cache)
```

Serialized IR will be stored in a scratch space, keyed on the cache name and any sharding keys.


### Stacked method tables

A utility `StackedMethodTable` is provided to facilitate layering multiple method tables:

```julia

MyMethodTableStack(world) = StackedMethodTable(world, overlay_table, base_table)

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
end
CC.method_table(interp::CustomInterpreter) = MyMethodTableStack(interp.world)
```
