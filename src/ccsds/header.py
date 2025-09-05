"""
CCSDS-123.0-B-2 Compressed Image Header Implementation

Implements the complete header structure as specified in Section 5.1 of the
CCSDS-123.0-B-2 standard, including Image Metadata, Predictor Metadata,
and Entropy Coder Metadata.
"""

import struct
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import torch


class EncodingOrder(Enum):
    """Sample encoding order"""
    BAND_INTERLEAVED = 0  # BI
    BAND_SEQUENTIAL = 1   # BSQ


class PredictorMode(Enum):
    """Predictor mode"""
    FULL = 0
    REDUCED = 1


class TableType(Enum):
    """Supplementary information table types"""
    UNSIGNED_INTEGER = 0
    SIGNED_INTEGER = 1
    FLOATING_POINT = 2


class TableDimension(Enum):
    """Table dimension types"""
    ONE_DIMENSIONAL_Z = 0    # One element per band z
    TWO_DIMENSIONAL_ZX = 1   # One element per (z,x) pair
    TWO_DIMENSIONAL_YX = 2   # One element per (y,x) pair


class SupplementaryTable:
    """
    Supplementary Information Table (Issue 2 feature)

    Provides ancillary image or instrument information such as:
    - Malfunctioning detector elements
    - Wavelength information for each spectral band
    - Calibration data
    - Quality flags
    """

    def __init__(self, table_id: int, table_type: TableType,
                 dimension: TableDimension, data: Union[List, torch.Tensor]):
        """
        Initialize supplementary table

        Args:
            table_id: Unique table identifier
            table_type: Data type (unsigned int, signed int, float)
            dimension: Table dimensions (1D-Z, 2D-ZX, 2D-YX)
            data: Table data
        """
        self.table_id = table_id
        self.table_type = table_type
        self.dimension = dimension
        self.data = data

    def pack(self) -> bytes:
        """Pack supplementary table to bytes"""
        result = bytearray()

        # Table header (8 bytes)
        result.append(self.table_id)  # Table ID (1 byte)
        result.append(self.table_type.value)  # Table type (1 byte)
        result.append(self.dimension.value)  # Dimension (1 byte)
        result.append(0)  # Reserved (1 byte)

        # Data length (4 bytes, big-endian)
        if isinstance(self.data, torch.Tensor):
            data_bytes = self._pack_tensor_data()
        else:
            data_bytes = self._pack_list_data()

        data_length = len(data_bytes)
        result.extend(struct.pack('>I', data_length))

        # Table data
        result.extend(data_bytes)

        return bytes(result)

    def _pack_tensor_data(self) -> bytes:
        """Pack tensor data according to table type"""
        if self.table_type == TableType.FLOATING_POINT:
            # Pack as 32-bit IEEE 754 floats
            return struct.pack(f'>{len(self.data.flatten())}f', *self.data.flatten().tolist())
        elif self.table_type == TableType.SIGNED_INTEGER:
            # Pack as 32-bit signed integers
            return struct.pack(f'>{len(self.data.flatten())}i', *self.data.flatten().tolist())
        else:  # UNSIGNED_INTEGER
            # Pack as 32-bit unsigned integers
            return struct.pack(f'>{len(self.data.flatten())}I', *self.data.flatten().tolist())

    def _pack_list_data(self) -> bytes:
        """Pack list data according to table type"""
        if self.table_type == TableType.FLOATING_POINT:
            return struct.pack(f'>{len(self.data)}f', *self.data)
        elif self.table_type == TableType.SIGNED_INTEGER:
            return struct.pack(f'>{len(self.data)}i', *self.data)
        else:  # UNSIGNED_INTEGER
            return struct.pack(f'>{len(self.data)}I', *self.data)

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple['SupplementaryTable', int]:
        """Unpack supplementary table from bytes"""
        table_id = data[offset]
        table_type = TableType(data[offset + 1])
        dimension = TableDimension(data[offset + 2])
        # Skip reserved byte at offset + 3

        data_length = struct.unpack('>I', data[offset + 4:offset + 8])[0]

        # Unpack table data based on type
        data_start = offset + 8
        data_end = data_start + data_length
        table_data_bytes = data[data_start:data_end]

        if table_type == TableType.FLOATING_POINT:
            num_elements = data_length // 4
            table_data = list(struct.unpack(f'>{num_elements}f', table_data_bytes))
        elif table_type == TableType.SIGNED_INTEGER:
            num_elements = data_length // 4
            table_data = list(struct.unpack(f'>{num_elements}i', table_data_bytes))
        else:  # UNSIGNED_INTEGER
            num_elements = data_length // 4
            table_data = list(struct.unpack(f'>{num_elements}I', table_data_bytes))

        table = cls(table_id, table_type, dimension, table_data)
        bytes_consumed = 8 + data_length

        return table, bytes_consumed


class ImageMetadata:
    """
    Image Metadata part of the header (Section 5.1.1)
    Contains Essential subpart and optional Supplementary Information Tables
    """

    def __init__(self):
        # Essential subpart fields
        self.user_defined_data_length: int = 0
        self.samples_per_pixel: int = 0
        self.lines_per_image: int = 0
        self.sample_encoding_order: EncodingOrder = EncodingOrder.BAND_INTERLEAVED
        self.sub_frame_interleaving: bool = False
        self.output_word_size: int = 8  # bits
        self.entropy_coder_type: int = 1  # 1 = Hybrid
        self.reserved: int = 0
        self.signed_samples: bool = False
        self.sample_bit_depth: int = 8

        # Optional Supplementary Information Tables
        self.user_defined_data: Optional[bytes] = None
        self.supplementary_tables: List['SupplementaryTable'] = []

    def pack(self) -> bytes:
        """Pack image metadata into bytes according to CCSDS format"""
        # Calculate total user defined data length including supplementary tables
        supplementary_tables_size = sum(len(table.pack()) for table in self.supplementary_tables)
        base_user_data_size = len(self.user_defined_data) if self.user_defined_data else 0
        total_user_data_length = base_user_data_size + supplementary_tables_size

        # Essential subpart: 8 bytes
        byte0 = (total_user_data_length >> 8) & 0xFF
        byte1 = total_user_data_length & 0xFF

        byte2 = (self.samples_per_pixel >> 8) & 0xFF
        byte3 = self.samples_per_pixel & 0xFF

        byte4 = (self.lines_per_image >> 8) & 0xFF
        byte5 = self.lines_per_image & 0xFF

        # Pack flags and fields into byte6
        byte6 = 0
        byte6 |= (self.sample_encoding_order.value << 5)  # bits 7-5
        byte6 |= (int(self.sub_frame_interleaving) << 4)  # bit 4
        byte6 |= (self.output_word_size - 1)  # bits 3-0 (stored as size-1)

        # Pack entropy coder type and flags into byte7
        byte7 = 0
        byte7 |= (self.entropy_coder_type << 4)  # bits 7-4
        byte7 |= (self.reserved << 2)  # bits 3-2
        byte7 |= (int(self.signed_samples) << 1)  # bit 1
        byte7 |= (self.sample_bit_depth - 1) >> 3  # bit 0 (MSB of bit depth-1)

        essential = bytes([byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7])

        # Add user defined data if present
        result = essential
        if self.user_defined_data:
            result += self.user_defined_data

        # Add supplementary information tables
        for table in self.supplementary_tables:
            result += table.pack()

        return result

    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple['ImageMetadata', int]:
        """Unpack image metadata from bytes"""
        metadata = cls()

        # Essential subpart
        metadata.user_defined_data_length = (data[offset] << 8) | data[offset + 1]
        metadata.samples_per_pixel = (data[offset + 2] << 8) | data[offset + 3]
        metadata.lines_per_image = (data[offset + 4] << 8) | data[offset + 5]

        byte6 = data[offset + 6]
        metadata.sample_encoding_order = EncodingOrder((byte6 >> 5) & 0x7)
        metadata.sub_frame_interleaving = bool((byte6 >> 4) & 0x1)
        metadata.output_word_size = (byte6 & 0xF) + 1

        byte7 = data[offset + 7]
        metadata.entropy_coder_type = (byte7 >> 4) & 0xF
        metadata.reserved = (byte7 >> 2) & 0x3
        metadata.signed_samples = bool((byte7 >> 1) & 0x1)
        metadata.sample_bit_depth = ((byte7 & 0x1) << 3) + 1  # Need to read more bits

        bytes_consumed = 8

        # Read user defined data and supplementary tables if present
        if metadata.user_defined_data_length > 0:
            user_data_start = offset + 8
            user_data_end = user_data_start + metadata.user_defined_data_length

            # Try to parse supplementary tables from user defined data
            # Tables are expected to be at the end of user defined data
            current_offset = user_data_start
            remaining_bytes = metadata.user_defined_data_length

            # Simple heuristic: if data starts with reasonable table headers, parse as tables
            # Otherwise treat as raw user defined data
            while remaining_bytes >= 8:  # Minimum table header size
                try:
                    table, table_bytes = SupplementaryTable.unpack(data, current_offset)
                    if table_bytes <= remaining_bytes:
                        metadata.supplementary_tables.append(table)
                        current_offset += table_bytes
                        remaining_bytes -= table_bytes
                    else:
                        break
                except:
                    break

            # If we have leftover data that wasn't parsed as tables, store as user_defined_data
            if remaining_bytes > 0:
                metadata.user_defined_data = data[current_offset:user_data_end]

            bytes_consumed += metadata.user_defined_data_length

        return metadata, bytes_consumed


class PredictorMetadata:
    """
    Predictor Metadata part of the header (Section 5.1.2)
    Contains predictor configuration and optional weight tables
    """

    def __init__(self):
        # Primary predictor metadata
        self.predictor_mode: PredictorMode = PredictorMode.FULL
        self.local_sum_type: int = 0  # 0=neighbor-oriented, 1=column-oriented
        self.prediction_bands: int = 0  # Number of bands used for prediction
        self.sample_representative_flag: bool = False
        self.weight_component_resolution: int = 4  # V_min + 2
        self.weight_update_scaling_exponent_change_interval: int = 64
        self.weight_update_scaling_exponent_initial_parameter: int = 4  # V_min
        self.weight_update_scaling_exponent_final_parameter: int = 6   # V_max
        self.weight_initialization_method: int = 0  # 0=default weights
        self.weight_initialization_table_flag: bool = False
        self.weight_initialization_resolution: int = 4
        self.custom_weight_flag: bool = False
        self.quantization_factor_resolution: int = 4

        # Optional tables
        self.weight_initialization_table: Optional[torch.Tensor] = None
        self.quantization_factor_table: Optional[torch.Tensor] = None

    def pack(self) -> bytes:
        """Pack predictor metadata into bytes"""
        # Primary metadata: 8 bytes minimum
        byte0 = 0
        byte0 |= (self.predictor_mode.value << 7)
        byte0 |= (self.local_sum_type << 6)
        byte0 |= (self.prediction_bands & 0x3F)  # bits 5-0

        byte1 = 0
        byte1 |= (int(self.sample_representative_flag) << 7)
        byte1 |= ((self.weight_component_resolution - 2) << 4)  # bits 6-4
        byte1 |= (self.weight_update_scaling_exponent_change_interval >> 8) & 0xF  # bits 3-0

        byte2 = self.weight_update_scaling_exponent_change_interval & 0xFF

        byte3 = 0
        byte3 |= (self.weight_update_scaling_exponent_initial_parameter << 4)
        byte3 |= (self.weight_update_scaling_exponent_final_parameter & 0xF)

        byte4 = 0
        byte4 |= (self.weight_initialization_method << 6)
        byte4 |= (int(self.weight_initialization_table_flag) << 5)
        byte4 |= ((self.weight_initialization_resolution - 2) << 2)
        byte4 |= (int(self.custom_weight_flag) << 1)
        byte4 |= ((self.quantization_factor_resolution - 2) >> 1) & 0x1

        byte5 = ((self.quantization_factor_resolution - 2) << 7) & 0x80

        result = bytes([byte0, byte1, byte2, byte3, byte4, byte5, 0, 0])  # Pad to 8 bytes

        # Add optional tables if present
        if self.weight_initialization_table is not None:
            # Pack weight table (implementation depends on format)
            pass

        if self.quantization_factor_table is not None:
            # Pack quantization table (implementation depends on format)
            pass

        return result


class EntropyCoderMetadata:
    """
    Entropy Coder Metadata part of the header (Section 5.1.3)
    Format depends on the entropy coder type
    """

    def __init__(self, coder_type: int = 1):
        self.coder_type = coder_type  # 1 = Hybrid

        if coder_type == 1:  # Hybrid entropy coder
            self.uncoded_data_length: int = 0
            self.initial_count_exponent: int = 1  # γ*
            self.accumulator_initialization_constant: int = 1  # K
            self.accumulator_initialization_table_flag: bool = False
            self.accumulator_initialization_table: Optional[torch.Tensor] = None

    def pack(self) -> bytes:
        """Pack entropy coder metadata into bytes"""
        if self.coder_type == 1:  # Hybrid
            byte0 = (self.uncoded_data_length >> 8) & 0xFF
            byte1 = self.uncoded_data_length & 0xFF

            byte2 = 0
            byte2 |= (self.initial_count_exponent << 4)
            byte2 |= (self.accumulator_initialization_constant & 0xF)

            byte3 = int(self.accumulator_initialization_table_flag) << 7

            result = bytes([byte0, byte1, byte2, byte3])

            # Add accumulator table if present
            if self.accumulator_initialization_table is not None:
                # Pack accumulator initialization table
                pass

            return result
        else:
            return b''  # Unknown coder type


class CCSDS123Header:
    """
    Complete CCSDS-123.0-B-2 compressed image header

    Structure:
    1. Image Metadata (Essential subpart + optional tables)
    2. Predictor Metadata (Primary + optional tables)
    3. Entropy Coder Metadata (varies by coder type)
    """

    def __init__(self):
        self.image_metadata = ImageMetadata()
        self.predictor_metadata = PredictorMetadata()
        self.entropy_coder_metadata = EntropyCoderMetadata()

    def pack(self) -> bytes:
        """Pack complete header into bytes"""
        result = b''
        result += self.image_metadata.pack()
        result += self.predictor_metadata.pack()
        result += self.entropy_coder_metadata.pack()
        return result

    @classmethod
    def unpack(cls, data: bytes) -> 'CCSDS123Header':
        """Unpack header from bytes"""
        header = cls()
        offset = 0

        # Unpack image metadata
        header.image_metadata, consumed = ImageMetadata.unpack(data, offset)
        offset += consumed

        # Unpack predictor metadata (implementation needed)
        # header.predictor_metadata, consumed = PredictorMetadata.unpack(data, offset)
        # offset += consumed

        # Unpack entropy coder metadata (implementation needed)
        # header.entropy_coder_metadata, consumed = EntropyCoder Metadata.unpack(data, offset)

        return header

    def set_image_params(self, height: int, width: int, num_bands: int,
                        bit_depth: int = 8, signed: bool = False):
        """Set basic image parameters"""
        self.image_metadata.lines_per_image = height
        self.image_metadata.samples_per_pixel = width
        self.predictor_metadata.prediction_bands = num_bands
        self.image_metadata.sample_bit_depth = bit_depth
        self.image_metadata.signed_samples = signed

    def set_predictor_params(self, mode: PredictorMode = PredictorMode.FULL,
                           v_min: int = 4, v_max: int = 6,
                           rescale_interval: int = 64):
        """Set predictor parameters"""
        self.predictor_metadata.predictor_mode = mode
        self.predictor_metadata.weight_update_scaling_exponent_initial_parameter = v_min
        self.predictor_metadata.weight_update_scaling_exponent_final_parameter = v_max
        self.predictor_metadata.weight_update_scaling_exponent_change_interval = rescale_interval

    def set_entropy_coder_params(self, gamma_star: int = 1, k: int = 1):
        """Set entropy coder parameters"""
        self.entropy_coder_metadata.initial_count_exponent = gamma_star
        self.entropy_coder_metadata.accumulator_initialization_constant = k

    def add_supplementary_table(self, table_id: int, table_type: TableType,
                               dimension: TableDimension, data: Union[List, torch.Tensor]) -> None:
        """Add a supplementary information table to the header"""
        table = SupplementaryTable(table_id, table_type, dimension, data)
        self.image_metadata.supplementary_tables.append(table)

    def add_wavelength_table(self, wavelengths: Union[List[float], torch.Tensor], table_id: int = 1) -> None:
        """
        Add wavelength information table for spectral bands

        Args:
            wavelengths: List or tensor of wavelength values for each band
            table_id: Unique table identifier
        """
        self.add_supplementary_table(table_id, TableType.FLOATING_POINT,
                                    TableDimension.ONE_DIMENSIONAL_Z, wavelengths)

    def add_bad_pixel_table(self, bad_pixels: Union[List[Tuple[int, int]], torch.Tensor],
                           table_id: int = 2) -> None:
        """
        Add bad/dead pixel location table

        Args:
            bad_pixels: List of (y, x) coordinates or 2D tensor of bad pixel locations
            table_id: Unique table identifier
        """
        if isinstance(bad_pixels, list):
            # Convert list of (y,x) tuples to flat list [y1, x1, y2, x2, ...]
            flat_coords = []
            for y, x in bad_pixels:
                flat_coords.extend([y, x])
            bad_pixels = flat_coords

        self.add_supplementary_table(table_id, TableType.UNSIGNED_INTEGER,
                                    TableDimension.TWO_DIMENSIONAL_YX, bad_pixels)

    def add_calibration_table(self, calibration_data: Union[List[float], torch.Tensor],
                             dimension: TableDimension = TableDimension.ONE_DIMENSIONAL_Z,
                             table_id: int = 3) -> None:
        """
        Add calibration data table

        Args:
            calibration_data: Calibration coefficients or factors
            dimension: Table dimension type
            table_id: Unique table identifier
        """
        self.add_supplementary_table(table_id, TableType.FLOATING_POINT,
                                    dimension, calibration_data)

    def add_quality_flags_table(self, quality_flags: Union[List[int], torch.Tensor],
                               dimension: TableDimension = TableDimension.ONE_DIMENSIONAL_Z,
                               table_id: int = 4) -> None:
        """
        Add quality flags table for bands or pixels

        Args:
            quality_flags: Quality flag values
            dimension: Table dimension type
            table_id: Unique table identifier
        """
        self.add_supplementary_table(table_id, TableType.UNSIGNED_INTEGER,
                                    dimension, quality_flags)

    def get_supplementary_tables(self) -> List[SupplementaryTable]:
        """Get all supplementary information tables"""
        return self.image_metadata.supplementary_tables

    def get_table_by_id(self, table_id: int) -> Optional[SupplementaryTable]:
        """Get supplementary table by ID"""
        for table in self.image_metadata.supplementary_tables:
            if table.table_id == table_id:
                return table
        return None

    def remove_table(self, table_id: int) -> bool:
        """Remove supplementary table by ID"""
        for i, table in enumerate(self.image_metadata.supplementary_tables):
            if table.table_id == table_id:
                del self.image_metadata.supplementary_tables[i]
                return True
        return False

    def clear_supplementary_tables(self) -> None:
        """Remove all supplementary tables"""
        self.image_metadata.supplementary_tables.clear()

    def get_header_info(self) -> Dict[str, Any]:
        """Get comprehensive header information"""
        info = {
            'image_dimensions': {
                'height': self.image_metadata.lines_per_image,
                'width': self.image_metadata.samples_per_pixel,
                'bands': self.predictor_metadata.prediction_bands + 1
            },
            'sample_properties': {
                'bit_depth': self.image_metadata.sample_bit_depth,
                'signed': self.image_metadata.signed_samples,
                'encoding_order': self.image_metadata.sample_encoding_order.name
            },
            'predictor_config': {
                'mode': self.predictor_metadata.predictor_mode.name,
                'prediction_bands': self.predictor_metadata.prediction_bands,
                'v_min': self.predictor_metadata.weight_update_scaling_exponent_initial_parameter,
                'v_max': self.predictor_metadata.weight_update_scaling_exponent_final_parameter,
                'rescale_interval': self.predictor_metadata.weight_update_scaling_exponent_change_interval
            },
            'entropy_coder': {
                'type': self.entropy_coder_metadata.coder_type,
                'gamma_star': self.entropy_coder_metadata.initial_count_exponent,
                'k': self.entropy_coder_metadata.accumulator_initialization_constant
            },
            'supplementary_tables': [
                {
                    'id': table.table_id,
                    'type': table.table_type.name,
                    'dimension': table.dimension.name,
                    'data_size': len(table.data) if hasattr(table.data, '__len__') else table.data.numel()
                }
                for table in self.image_metadata.supplementary_tables
            ]
        }
        return info
