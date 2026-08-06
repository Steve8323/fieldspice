# Security Policy

## Scope

fieldspice is a scientific simulation library. It does not handle credentials,
network traffic, or untrusted remote input in normal use. The realistic risk
surface is small but not empty:

* `Netlist.from_spice()` parses text. Malformed input should raise, not execute.
* `io.load_result()` / `io.load_grid()` read files. They use HDF5 or NumPy
  `.npz` with `allow_pickle=False`; loading a file from an untrusted source
  should never execute code. If you find a path where it can, that is a
  vulnerability and I want to hear about it.

## Reporting

Email **contact.levu@proton.me**, or open a private security advisory through
GitHub's "Report a vulnerability" button on the Security tab.

Please include a reproduction. I will acknowledge within a week.

## Supported versions

Only the latest release is supported. The project is alpha (0.x) and the API
is not yet stable.
