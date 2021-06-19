import { Program, VertexBuffer, IndexBuffer } from "./utils";
import { BaseGLGlyph, Transform } from "./base";
import { ScatterView } from "../scatter";
import { CircleView } from "../circle";
import { MarkerType } from "../../../core/enums";
declare type MarkerLikeView = ScatterView | CircleView;
export declare class MarkerGL extends BaseGLGlyph {
    readonly glyph: MarkerLikeView;
    readonly marker_type: MarkerType;
    protected prog: Program;
    protected vbo_sx: VertexBuffer;
    protected vbo_sy: VertexBuffer;
    protected vbo_s: VertexBuffer;
    protected vbo_a: VertexBuffer;
    protected vbo_linewidth: VertexBuffer & {
        used?: boolean;
    };
    protected vbo_fg_color: VertexBuffer & {
        used?: boolean;
    };
    protected vbo_bg_color: VertexBuffer & {
        used?: boolean;
    };
    protected index_buffer: IndexBuffer;
    static is_supported(marker_type: MarkerType): boolean;
    constructor(gl: WebGLRenderingContext, glyph: MarkerLikeView, marker_type: MarkerType);
    draw(indices: number[], main_glyph: MarkerLikeView, trans: Transform): void;
    protected _set_data(nvertices: number): void;
    protected _set_visuals(nvertices: number): void;
}
export {};
//# sourceMappingURL=markers.d.ts.map