import { Scale } from "./scale";
import { map, left_edge_index } from "../../core/util/arrayable";
export class LinearInterpolationScale extends Scale {
    constructor(attrs) {
        super(attrs);
    }
    static init_LinearInterpolationScale() {
        this.internal(({ Arrayable }) => ({
            binning: [Arrayable],
        }));
    }
    get s_compute() {
        throw new Error("not implemented");
    }
    compute(x) {
        return x;
    }
    v_compute(vs) {
        const { binning } = this;
        const { start, end } = this.source_range;
        const min_val = start;
        const max_val = end;
        const n = binning.length;
        const step = (end - start) / (n - 1);
        const mapping = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            mapping[i] = start + i * step;
        }
        const vvs = map(vs, (v) => {
            if (v < min_val)
                return min_val;
            if (v > max_val)
                return max_val;
            const k = left_edge_index(v, binning);
            const b0 = binning[k];
            const b1 = binning[k + 1];
            const c = (v - b0) / (b1 - b0);
            const m0 = mapping[k];
            const m1 = mapping[k + 1];
            return m0 + c * (m1 - m0);
        });
        return this._linear_v_compute(vvs);
    }
    invert(xprime) {
        return xprime;
    }
    v_invert(xprimes) {
        return new Float64Array(xprimes);
    }
}
LinearInterpolationScale.__name__ = "LinearInterpolationScale";
LinearInterpolationScale.init_LinearInterpolationScale();
//# sourceMappingURL=linear_interpolation_scale.js.map