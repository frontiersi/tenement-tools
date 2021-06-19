import { ScalarExpression } from "./expression";
import { min } from "../../core/util/array";
export class Minimum extends ScalarExpression {
    constructor(attrs) {
        super(attrs);
    }
    static init_Minimum() {
        this.define(({ Number, String, Nullable }) => ({
            field: [String],
            initial: [Nullable(Number), null],
        }));
    }
    _compute(source) {
        const column = source.data[this.field] ?? [];
        return Math.min(this.initial ?? Infinity, min(column));
    }
}
Minimum.__name__ = "Minimum";
Minimum.init_Minimum();
//# sourceMappingURL=minimum.js.map