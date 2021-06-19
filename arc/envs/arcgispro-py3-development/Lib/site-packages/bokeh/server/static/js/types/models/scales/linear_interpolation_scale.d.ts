import { Scale } from "./scale";
import { Arrayable, ScreenArray, FloatArray } from "../../core/types";
import * as p from "../../core/properties";
export declare namespace LinearInterpolationScale {
    type Attrs = p.AttrsOf<Props>;
    type Props = Scale.Props & {
        binning: p.Property<Arrayable<number>>;
    };
}
export interface LinearInterpolationScale extends LinearInterpolationScale.Attrs {
}
export declare class LinearInterpolationScale extends Scale {
    properties: LinearInterpolationScale.Props;
    constructor(attrs?: Partial<LinearInterpolationScale.Attrs>);
    static init_LinearInterpolationScale(): void;
    get s_compute(): (x: number) => number;
    compute(x: number): number;
    v_compute(vs: Arrayable<number>): ScreenArray;
    invert(xprime: number): number;
    v_invert(xprimes: Arrayable<number>): FloatArray;
}
//# sourceMappingURL=linear_interpolation_scale.d.ts.map