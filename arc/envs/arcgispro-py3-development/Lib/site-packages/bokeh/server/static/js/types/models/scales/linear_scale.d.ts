import { ContinuousScale } from "./continuous_scale";
import { Arrayable, ScreenArray, FloatArray } from "../../core/types";
import * as p from "../../core/properties";
export declare namespace LinearScale {
    type Attrs = p.AttrsOf<Props>;
    type Props = ContinuousScale.Props;
}
export interface LinearScale extends LinearScale.Attrs {
}
export declare class LinearScale extends ContinuousScale {
    properties: LinearScale.Props;
    constructor(attrs?: Partial<LinearScale.Attrs>);
    get s_compute(): (x: number) => number;
    compute(x: number): number;
    v_compute(xs: Arrayable<number>): ScreenArray;
    invert(xprime: number): number;
    v_invert(xprimes: Arrayable<number>): FloatArray;
}
//# sourceMappingURL=linear_scale.d.ts.map