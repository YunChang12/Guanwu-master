from __future__ import annotations


class BlueBirdError(Exception):
    """Base exception for all BlueBird errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ConfigError(BlueBirdError):
    ...


class AccessError(BlueBirdError):
    ...


class FetchError(BlueBirdError):
    ...


class ChecksumError(BlueBirdError):
    ...


class ParseError(BlueBirdError):
    ...


class NormalizeError(BlueBirdError):
    ...


class ValidationError(BlueBirdError):
    ...


class LicensePolicyError(BlueBirdError):
    ...
