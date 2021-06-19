import { Filter } from "./filter";
import { Indices } from "../../core/types";
export class BooleanFilter extends Filter {
    constructor(attrs) {
        super(attrs);
    }
    static init_BooleanFilter() {
        this.define(({ Boolean, Array, Nullable }) => ({
            booleans: [Nullable(Array(Boolean)), null],
        }));
    }
    compute_indices(source) {
        const size = source.length;
        const { booleans } = this;
        if (booleans == null) {
            return Indices.all_set(size);
        }
        else {
            return Indices.from_booleans(size, booleans);
        }
    }
}
BooleanFilter.__name__ = "BooleanFilter";
BooleanFilter.init_BooleanFilter();
//# sourceMappingURL=boolean_filter.js.map