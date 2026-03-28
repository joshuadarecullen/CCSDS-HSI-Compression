"""
CCSDS-123.0-B-2 I/O Components

This module contains I/O and formatting components:
- CCSDS123Header: Compressed image header
- SampleIterator: Sample encoding order iteration
- EncodingOrderOptimizer: Encoding order optimization
"""

from .header import (
    CCSDS123Header,
    PredictorMode,
    EncodingOrder,
    SupplementaryTable,
    TableType,
    TableDimension
)

from .encoding_orders import (
    SampleIterator,
    EncodingOrderOptimizer,
    EncodingOrder as SampleEncodingOrder
)

__all__ = [
    'CCSDS123Header',
    'PredictorMode',
    'EncodingOrder',
    'SupplementaryTable',
    'TableType',
    'TableDimension',
    'SampleIterator',
    'EncodingOrderOptimizer',
    'SampleEncodingOrder'
]
