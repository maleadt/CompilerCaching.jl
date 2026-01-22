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
with a method instance and three callbacks:
- `emit_ir`: Analyze source code and emit high-level IR, while populating the cache with any dependent code instances. Often this boils down to invoking Julia's inference engine.
- `emit_code`: Generate serializable code that can be cached across sessions (e.g. LLVM IR).
- `emit_executable`: Generate a session-specific representation (e.g., a function pointer).

Or, in code:
```julia
using CompilerCaching

# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheHandle
    world::UInt
    ...
end
@setup_caching CustomInterpreter.cache

# Let Julia populate the cache and return codeinfos for emit_code
function emit_ir(cache, mi, world)
    interp = CustomInterpreter(cache, world)
    return CompilerCaching.populate!(cache, interp, mi)
end

# generate some code representation
function emit_code(cache, mi, world, ir) end

# compile code to function pointer
function emit_executable(cache, mi, world, code) end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world),
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    ptr = cached_compilation(cache, mi, world; emit_ir, emit_code, emit_executable)
    ccall(ptr, ...)
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
    cached_compilation(cache, mi, world; emit_ir, emit_code, emit_executable)
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
                    # if needed, look for global methods too
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    cached_compilation(cache, mi, world; emit_ir, emit_code, emit_executable)
end
```

### Foreign IR

For compilers that define their own IR format that Julia doesn't know about, we cannot rely
on inference to populate the cache. Instead of calling `populate!`, we need to analyze our
IR and call `cache!` for each method instance, providing an array of dependencies to ensure
backedges are correctly set-up:

```julia
Base.Experimental.@MethodTable method_table

# Only define our special functions in the overlay method table,
# providing our custom IR as the source.
function really_special end
add_method(method_table, really_special, (Int,), MyCustomIR([:a, :b]))

# Since we're not relying on Julia's inference, we need to create and cache our own CIs.
# Recursive invocation of the underlying `get_ir` ensures dependencies are cached too.
function emit_ir(cache, mi, world)
    ir = mi.def.source::MyCustomIR
    deps = Core.MethodInstance[]
    for callee in ir.callees
        callee_mi = method_instance(callee.f, callee.tt; world, method_table)
        callee_ci, callee_ir = get_ir(cache, callee_mi, world; emit_ir)
        # do something with the callee's IR if needed
        push!(deps, callee_mi)
    end
    cache!(cache, mi; world, deps)
    return ir
end

function emit_code(cache, mi, world, ir) end

function emit_executable(cache, mi, world, code) end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheHandle(:MyCompiler)
    cached_compilation(cache, mi, world; emit_ir, emit_code, emit_executable)
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
