import { ID, ByteOrder } from "../types";
import { DataType, NDArray } from "./ndarray";
export declare type Shape = number[];
export declare type BufferRef = {
    __buffer__: string;
    order: ByteOrder;
    dtype: DataType;
    shape: Shape;
};
export declare type NDArrayRef = {
    __ndarray__: string | {
        toJSON(): string;
    };
    order: ByteOrder;
    dtype: DataType;
    shape: Shape;
};
export declare function is_NDArray_ref(v: unknown): v is BufferRef | NDArrayRef;
export declare type Buffers = Map<ID, ArrayBuffer>;
export declare function decode_NDArray(ref: BufferRef | NDArrayRef, buffers: Buffers): NDArray;
export declare function encode_NDArray(array: NDArray, buffers?: Buffers): BufferRef | NDArrayRef;
//# sourceMappingURL=serialization.d.ts.map