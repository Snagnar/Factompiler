"""Intermediate representation package for the Factorio Circuit DSL."""

from .builder import IRBuilder
from .nodes import (
    IRNode,
    IRValue,
    IREffect,
    IR_Arith,
    IR_Const,
    IR_Decider,
    IR_EntityPropRead,
    IR_EntityPropWrite,
    IR_FuncCall,
    IR_FuncDecl,
    IR_Group,
    IR_MemCreate,
    IR_MemRead,
    IR_MemWrite,
    IR_PlaceEntity,
    IR_WireMerge,
    IR_ConnectToWire,
    SignalRef,
    ValueRef,
)
from .optimizer import CSEOptimizer

__all__ = [
    "SignalRef",
    "ValueRef",
    "IRNode",
    "IRValue",
    "IREffect",
    "IR_Const",
    "IR_Arith",
    "IR_Decider",
    "IR_WireMerge",
    "IR_MemRead",
    "IR_EntityPropRead",
    "IR_EntityPropWrite",
    "IR_MemCreate",
    "IR_MemWrite",
    "IR_PlaceEntity",
    "IR_ConnectToWire",
    "IR_Group",
    "IR_FuncDecl",
    "IR_FuncCall",
    "IRBuilder",
    "CSEOptimizer",
]
