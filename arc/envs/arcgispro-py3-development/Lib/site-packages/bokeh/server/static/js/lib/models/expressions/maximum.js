import { ScalarExpression } from "./expression";
import { max } from "../../core/util/array";
export class Maximum extends ScalarExpression {
    constructor(attrs) {
        super(attrs);
    }
    static init_Maximum() {
        this.define(({ Number, String, Nullable }) => ({
            field: [String],
            initial: [Nullable(Number), null],
        }));
    }
    _compute(source) {
        const column = source.data[this.field] ?? [];
        return Math.max(this.initial ?? -Infinity, max(column));
    }
}
Maximum.__name__ = "Maximum";
Maximum.init_Maximum();
//# sourceMappingURL=maximum.js.map