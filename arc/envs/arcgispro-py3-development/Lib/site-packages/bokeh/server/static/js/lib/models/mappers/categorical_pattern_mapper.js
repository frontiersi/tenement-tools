import { cat_v_compute } from "./categorical_mapper";
import { FactorSeq } from "../ranges/factor_range";
import { Mapper } from "./mapper";
import { HatchPatternType } from "../../core/enums";
export class CategoricalPatternMapper extends Mapper {
    constructor(attrs) {
        super(attrs);
    }
    static init_CategoricalPatternMapper() {
        this.define(({ Number, Array, Nullable }) => ({
            factors: [FactorSeq],
            patterns: [Array(HatchPatternType)],
            start: [Number, 0],
            end: [Nullable(Number), null],
            default_value: [HatchPatternType, " "],
        }));
    }
    v_compute(xs) {
        const values = new Array(xs.length);
        cat_v_compute(xs, this.factors, this.patterns, values, this.start, this.end, this.default_value);
        return values;
    }
}
CategoricalPatternMapper.__name__ = "CategoricalPatternMapper";
CategoricalPatternMapper.init_CategoricalPatternMapper();
//# sourceMappingURL=categorical_pattern_mapper.js.map