import { VisualProperties, VisualUniforms } from "./visual";
import { uint32 } from "../types";
import * as p from "../properties";
import * as mixins from "../property_mixins";
import { Context2d } from "../util/canvas";
export interface Fill extends Readonly<mixins.Fill> {
}
export declare class Fill extends VisualProperties {
    get doit(): boolean;
    set_value(ctx: Context2d): void;
}
export declare class FillScalar extends VisualUniforms {
    readonly fill_color: p.UniformScalar<uint32>;
    readonly fill_alpha: p.UniformScalar<number>;
    get doit(): boolean;
    set_value(ctx: Context2d): void;
}
export declare class FillVector extends VisualUniforms {
    readonly fill_color: p.Uniform<uint32>;
    readonly fill_alpha: p.Uniform<number>;
    get doit(): boolean;
    set_vectorize(ctx: Context2d, i: number): void;
}
//# sourceMappingURL=fill.d.ts.map