import inspect

import pmwd

try:
    sig = inspect.signature(pmwd.Configuration)
    print(f"Configuration Signature: {sig}")
except Exception as e:
    print(f"Error: {e}")

try:
    # Check fields if it's a named tuple
    conf = pmwd.Configuration(ptcl_spacing=1.0, mesh_shape=(1,1,1))
    print(f"Fields: {conf._fields}")
except Exception as e:
    print(f"Error instantiating/inspecting: {e}")
