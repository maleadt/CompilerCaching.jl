# CompilerCaching.jl

A package for interfacing with Julia's compiler caching infrastructure for the purpose
of building custom compilers. It provides a typed view on the integrated code cache
(`Core.Compiler.InternalCodeCache`), drives inference into it, and attaches custom
compilation results to cached `CodeInstance`s — including across precompilation, so
that compilers building on this package can ship precompiled artifacts in package
images.

Requires Julia 1.11 or later. On older versions the package loads as an empty shell,
so that downstream packages can depend on it unconditionally and provide their own
fallback (e.g. GPUCompiler.jl, which retains a session-local cache on Julia 1.10).


## Installation

```julia
using Pkg
Pkg.add(url="path/to/CompilerCaching")
```

## Basic usage

Julia's code caches are indexed with method instances, yielding a code instance that keeps
track of compilation results. Code instances are owned by a specific compiler, identified
by an owner token, and they contain a cache of results specific to that compiler.

The basic usage pattern of working with the compiler cache through CompilerCaching.jl:
1. Define a mutable struct with a zero-arg constructor to hold compilation results
2. Create a `CacheView{V}(owner_token, world)` where `V` is your results struct type
3. Use the cache's `Dict` interface to get or create a code instance for a method instance
4. Access cached compilation results via `results(cache, ci)`, populating them if needed

```julia
using CompilerCaching

# Define your results struct
mutable struct MyResults
    executable::Any
    MyResults() = new(nothing)
end

# Compile a method instance
function compile!(cache, mi)
    # Get or create code instance
    ci = get!(cache, mi) do
        create_ci(cache, mi)
    end

    # Check for cache hit
    res = results(cache, ci)
    res.executable !== nothing && return res.executable

    # Generate an executable.
    # Use multiple steps (e.g. IR generation, machine code generation, linking) if needed.
    if res.executable === nothing
        res.executable = emit_executable(cache, mi, res.code)
    end

    return res.executable
end

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheView{MyResults}(:MyCompiler, world)
    exe = compile!(cache, mi)
    ccall(exe, ...)
end
```

The `create_ci` function creates a bare code instance with (initially empty)
compilation results. Most users will want to rely on Julia's type inference
to instead populate the cache with a code instance that knows about dependent
methods for invalidation purposes, and contains inferred source code for further
compilation. This can be done with a custom abstract interpreter and the `typeinf!`
function from this package:

```julia
# Set-up a custom interpreter, and link it to the cache
struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
    ...
end
CC.cache_owner(::CustomInterpreter) = :MyCompiler
CC.get_inference_world(interp::CustomInterpreter) = interp.world

function compile!(cache, mi)
    # Get CI through inference
    ci = get(cache, mi, nothing)
    if ci === nothing
        interp = CustomInterpreter(cache.world)
        ci = CompilerCaching.typeinf!(interp, mi)
    end

    # ... further compilation steps
end
```

Beyond the standard `AbstractInterpreter` interface, the interpreter only needs the
two methods Julia itself uses to address the integrated cache:
- `CC.cache_owner(interp)` — partitions the integrated cache (same-owner CIs share storage)
- `CC.get_inference_world(interp)` — world age inference runs at

There is no inference-time hook for results: `results(cache, ci)` attaches the
results struct on first access, so any interpreter whose owner matches the cache
view works out of the box.


## Cache sharding

It is possible to partition the cache by additional parameters by using a tuple or
named tuple as the owner key type:

```julia
function call(f, args...; opt_level=1)
    # ...

    cache = CacheView{MyResults}((:MyCompiler, opt_level), world)

    # ...
end
```

Different calls with the same owner key will hit the same cache partition.


## Overlay methods

It is often useful to redefine existing methods for use with the custom compiler.
This can be accomplished using overlay methods in a custom method table:

```julia
Base.Experimental.@MethodTable method_table
Base.Experimental.@overlay method_table function Base.sin(x::Int)
    # custom implementation
end

# Expose the method table to the interpreter
struct CustomInterpreter <: CC.AbstractInterpreter
    cache::CacheView
    ...
end
CC.method_table(interp::CustomInterpreter) = CC.OverlayMethodTable(interp.cache.world, method_table)

function call(f, args...)
    tt = map(Core.Typeof, args)
    world = Base.get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    # if needed, look for global methods too
                    throw(MethodError(f, args)))

    # ...
end
```

If multiple overlay tables are needed, they can be stacked using `StackedMethodTable`:

```julia
MyMethodTableStack(world) = StackedMethodTable(world, overlay_table, base_table)

struct CustomInterpreter <: CC.AbstractInterpreter
    world::UInt
end
CC.method_table(interp::CustomInterpreter) = MyMethodTableStack(interp.world)
```


## Foreign IR

For compilers that define their own IR format that Julia doesn't know about, we cannot rely
on inference to populate the cache, so we need to bring our own code instances using
`create_ci`:

```julia
Base.Experimental.@MethodTable method_table

# Results struct for foreign IR
mutable struct ForeignResults
    ir::Any
    ForeignResults() = new(nothing)
end

# Only define our special functions in the overlay method table,
# providing our custom IR as the source.
function really_special end
add_method(method_table, really_special, (Int,), MyCustomIR([:a, :b]))

# Compile function using get! do-block pattern
function compile!(cache, mi)
    ci = get!(cache, mi) do
        source = mi.def.source::MyCustomIR
        ir = infer(source)

        deps = Core.MethodInstance[]
        for callee in ir.callees
            callee_mi = method_instance(callee.f, callee.tt; world=cache.world, method_table)
            compile!(cache, callee_mi)  # recursive compilation
            push!(deps, callee_mi)
        end

        ci = create_ci(cache, mi; deps)
        results(cache, ci).ir = ir  # cache the inferred IR if needed
        return ci
    end

    # ...
end

function call(f, args...)
    tt = Tuple{map(Core.Typeof, args)...}
    world = get_world_counter()
    mi = @something(method_instance(f, tt; world, method_table),
                    throw(MethodError(f, args)))

    cache = CacheView{ForeignResults}(:MyCompiler, world)
    exe = compile!(cache, mi)
    ccall(exe, ...)
end
```


## Persistent artifacts

Compilation results handled by this package live in three tiers:

1. **Session**: the results struct attached to a `CodeInstance` (`results(cache, ci)`).
   Invalidated with the code instance.
2. **Package image**: the same struct, serialized when the owning package is precompiled.
   Covers whatever the package's precompile workload reaches.
3. **Object cache**: byte artifacts produced at run time (a CUBIN, a shared library, an
   object file), stored in Julia's on-disk object cache and reused by later sessions.

The third tier is exposed by `CompilerCaching.ObjCache`. On Julia 1.14 (development build
3075 or later) it uses the runtime's store, shared with the JIT and configured through the
same environment variables (`JULIA_OBJCACHE=0` disables, `JULIA_OBJCACHE_PATH` relocates,
`JULIA_OBJCACHE_CAPACITY` sizes it, default 512 MiB). On older Julia the package provides
an LMDB-backed port of that store (same layout, capacity accounting, eviction order and
environment variables), living in a Julia-minor-specific directory in CompilerCaching's
scratch space; LMDB is only loaded there.

```julia
using CompilerCaching: ObjCache

function compile_kernel(res, source::Vector{UInt8}, target::String)
    res.binary !== nothing && return res.binary  # session tier
    res.binary = ObjCache.get!("MyCompiler/binary", toolchain_version(), target, source;
                               schema=1, persistable=true) do
        run_toolchain(source, target)::Vector{UInt8}
    end
end
```

`get!` derives a SHA-256 key from the namespace, `schema` and the fields, returns the
stored bytes on a hit, and otherwise runs the block and submits its result for storage.
With `persistable=false` or no store, it simply runs the block. The store owns no policy;
consumers must honour this contract:

- **Key every input.** The fields must cover everything the value depends on, including
  the external toolchain's identity. Under-keying serves stale artifacts silently.
- **Write once.** The same key must always mean equivalent bytes.
- **Writes are best-effort.** Julia's runtime backend commits on a background thread and
  does not drain pending writes at exit; the fallback commits synchronously. Consumers
  must rely on the weaker runtime contract, so a process that compiles and exits
  immediately may not populate the store.
- **The store is per Julia minor version**, not per build. This partitions storage but
  does not make an under-keyed artifact valid across Julia builds.
- **Keep the artifact in your session-tier struct.** `get!` has no in-memory memo.
- **Values above about 32 MiB are dropped**, and the capacity is shared with the JIT under
  LRU eviction, so any entry may disappear.
- Only persist artifacts that are valid in another process under the given key. Anything
  embedding session-local addresses, or depending on compatibility inputs that cannot be
  keyed completely, must pass `persistable=false`.

Namespaces follow `"<PackageName>/<artifact-kind>"`. Lower-level access is available via
`ObjCache.enabled`, `ObjCache.get`, `ObjCache.put!` and `ObjCache.keyhash`.
