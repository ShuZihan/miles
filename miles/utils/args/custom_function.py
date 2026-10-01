import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import ConfigDict, SerializeAsAny, create_model, model_validator

from miles.utils.args.schema import BaseConfig, validate_complete_config
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import with_relax_parser_required_args, with_suppressed_parser_help


class CustomFunctionConfig(BaseConfig):
    model_config = ConfigDict(frozen=True)

    path: str
    config: SerializeAsAny[BaseConfig] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_config(cls, values: Any) -> Any:
        if not isinstance(values, dict) or not isinstance(values.get("config"), dict):
            return values
        fn = load_function(values["path"])
        config_class = _compute_config_class(fn, path=values["path"])
        assert config_class is not None
        return values | {"config": validate_complete_config(config_class, values["config"])}

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return _restore_custom_function_config, (self.model_dump(),)


def add_user_provided_function_arguments(
    parser: argparse.ArgumentParser,
    *,
    modify_args: Callable[[argparse.Namespace], None] | None = None,
    config_class: type[BaseConfig] | None = None,
) -> argparse.ArgumentParser:
    try:
        with with_relax_parser_required_args(parser), with_suppressed_parser_help(parser):
            args_partial, _ = parser.parse_known_args()
    except SystemExit:
        return parser

    if modify_args is not None:
        modify_args(args_partial)
    infos = _compute_custom_function_field_infos(args_partial, partial=True, config_class=config_class)
    registered_paths: set[str] = set()
    registered_config_classes: set[type[BaseConfig]] = set()
    for info in infos:
        if info.path in registered_paths:
            continue
        registered_paths.add(info.path)
        if (
            function_config_class := info.config_class
        ) is not None and function_config_class not in registered_config_classes:
            function_config_class.add_arguments(parser=parser)
            registered_config_classes.add(function_config_class)
    return parser


def resolve_custom_function_configs(args: argparse.Namespace, *, config_class: type[BaseConfig] | None = None) -> None:
    custom_arg_names: set[str] = set()
    for info in _compute_custom_function_field_infos(args, config_class=config_class):
        resolved = resolve_custom_function_config(path=info.path, args=args)
        if resolved.config is not None:
            custom_arg_names.update(_custom_config_field_names(type(resolved.config)))
        setattr(args, info.name, resolved)

    for name in custom_arg_names:
        if hasattr(args, name):  # config-access-exempt: schema-selected field
            delattr(args, name)


def resolve_custom_function_config(*, path: str, args: argparse.Namespace) -> CustomFunctionConfig:
    config = None
    if (config_class := _compute_config_class(load_function(path), path=path)) is not None:
        config = config_class.model_validate(
            {
                key: getattr(args, key) for key in _custom_config_field_names(config_class) if hasattr(args, key)
            }  # config-access-exempt: schema-selected fields
        )
    return CustomFunctionConfig(path=path, config=config)


def _custom_config_field_names(config_class: type[BaseConfig]) -> set[str]:
    field_names = set(config_class.model_fields)
    if issubclass(config_class, _LegacyCustomFunctionConfig):
        field_names.update(config_class._suppressed_argument_names)
    return field_names


@dataclass(frozen=True)
class _CustomFunctionFieldInfo:
    name: str
    path: str
    fn: Any
    config_class: type[BaseConfig] | None


def _compute_custom_function_field_infos(
    args: argparse.Namespace, *, partial: bool = False, config_class: type[BaseConfig] | None = None
) -> list[_CustomFunctionFieldInfo]:
    from miles.utils.args.runtime import AllConfig

    paths = [
        (
            name,
            getattr(args, name, None) if partial else getattr(args, name),
        )  # config-access-exempt: schema-selected field
        for name, field in (config_class or AllConfig).model_fields.items()
        if field.annotation in {CustomFunctionConfig, CustomFunctionConfig | None}
    ]
    infos = []
    for name, path in paths:
        if path is None:
            continue
        try:
            fn = load_function(path)
        except (ModuleNotFoundError, ValueError):
            if partial:
                continue
            raise
        if partial and fn is None:
            continue
        config_class = _compute_config_class(fn, path=path)
        infos.append(_CustomFunctionFieldInfo(name=name, path=path, fn=fn, config_class=config_class))
    return infos


def _compute_config_class(fn: Any, *, path: str) -> type[BaseConfig] | None:
    config_class = getattr(fn, "config_class", None)  # config-access-exempt: custom hook protocol discovery
    if config_class is not None:
        assert isinstance(config_class, type) and issubclass(
            config_class, BaseConfig
        ), f"{path}.config_class must inherit BaseConfig"
        return config_class

    add_arguments = getattr(fn, "add_arguments", None)  # config-access-exempt: legacy hook protocol discovery
    if callable(add_arguments):
        fn.config_class = _adapt_legacy_custom_config(add_arguments)
        return fn.config_class

    return None


def _adapt_legacy_custom_config(add_arguments: Callable[[argparse.ArgumentParser], Any]) -> type[BaseConfig]:
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)

    fields = {
        action.dest: (
            Any,
            ... if action.required else action.default,
        )
        for action in parser._actions
        if action.dest != argparse.SUPPRESS and (action.required or action.default != argparse.SUPPRESS)
    }
    config_class = create_model("LegacyCustomFunctionConfig", __base__=_LegacyCustomFunctionConfig, **fields)
    config_class._suppressed_argument_names = frozenset(
        action.dest
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
        and not action.required
        and action.default == argparse.SUPPRESS
        and action.dest not in fields
    )

    def _add_arguments(parser: argparse.ArgumentParser) -> Any:
        return add_arguments(parser)

    config_class.add_arguments = staticmethod(_add_arguments)
    return config_class


class _LegacyCustomFunctionConfig(BaseConfig):
    model_config = ConfigDict(extra="allow")

    _suppressed_argument_names: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _validate_argument_names(cls, values: Any) -> Any:
        if isinstance(values, Mapping):
            unknown = values.keys() - cls.model_fields.keys() - cls._suppressed_argument_names
            if unknown:
                raise ValueError(f"Unknown legacy custom configuration fields: {sorted(unknown)}")
        return values


def _restore_custom_function_config(values: dict[str, Any]) -> CustomFunctionConfig:
    return CustomFunctionConfig.model_validate(values)
