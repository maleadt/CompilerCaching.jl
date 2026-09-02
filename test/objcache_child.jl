using CompilerCaching
using Serialization
using Test

const OC = CompilerCaching.ObjCache
const NS = "CompilerCachingTests/objcache"

function poll_get(ns, key; timeout=5.0)
    t0 = time()
    while true
        value = OC.get(ns, key)
        value !== nothing && return value
        time() - t0 > timeout && return nothing
        sleep(0.01)
    end
end

# Entries are committed in order, so observing this later sentinel means that all
# earlier writes (or their absence) have settled.
function settle(token)
    key = OC.keyhash(0, "sentinel", token)
    OC.put!(NS, key, UInt8[0])
    poll_get(NS, key) !== nothing || error("sentinel never committed")
end

Core.eval(Main, deserialize(stdin))
