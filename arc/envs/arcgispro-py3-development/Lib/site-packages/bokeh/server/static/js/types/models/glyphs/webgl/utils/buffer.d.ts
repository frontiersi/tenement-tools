import { TypedArray } from "../../../../core/util/ndarray";
export declare abstract class Buffer {
    readonly gl: WebGLRenderingContext;
    protected _target: number;
    protected _usage: number;
    readonly handle: WebGLBuffer;
    buffer_size: number;
    constructor(gl: WebGLRenderingContext);
    delete(): void;
    activate(): void;
    deactivate(): void;
    set_size(nbytes: number): void;
    set_data(offset: number, data: TypedArray | Uint8ClampedArray): void;
}
export declare class VertexBuffer extends Buffer {
    _target: number;
}
export declare class IndexBuffer extends Buffer {
    _target: number;
}
//# sourceMappingURL=buffer.d.ts.map