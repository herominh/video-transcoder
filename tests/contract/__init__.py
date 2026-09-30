"""Transcode contract v2 (draft): the worker-side validator, test-only in B03.

The contract tree itself lives in tests/contracts/transcode/v2/ (a byte-identical
copy of the Hub's laravel/resources/contracts/transcode/v2/); its README.md is the
normative rule text this package implements. Nothing at run time imports this
package yet; B08 moves it into core/ when the worker speaks v2.
"""
