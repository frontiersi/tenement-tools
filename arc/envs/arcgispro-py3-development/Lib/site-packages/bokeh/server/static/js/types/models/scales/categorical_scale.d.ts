import { Scale } from "./scale";
import { FactorRange } from "../ranges/factor_range";
import { Arrayable, ScreenArray, FloatArray } from "../../core/types";
import * as p from "../../core/properties";
export declare namespace CategoricalScale {
    type Attrs = p.AttrsOf<Props>;
    type Props = Scale.Props;
}
export interface CategoricalScale extends CategoricalScale.Attrs {
}
export declare class CategoricalScale extends Scale {
    properties: CategoricalScale.Props;
    constructor(attrs?: Partial<CategoricalScale.Attrs>);
    source_range: FactorRange;
    get s_compute(): (x: number) => number;
    compute(x: any): number;
    v_compute(xs: Arrayable<any>): ScreenArray;
    invert(xprime: number): number;
    v_invert(xprimes: Arrayable<number>): FloatArray;
}
//# sourceMappingURL=categorical_scale.d.ts.map