import { Expression } from "./expression";
export class Stack extends Expression {
    constructor(attrs) {
        super(attrs);
    }
    static init_Stack() {
        this.define(({ String, Array }) => ({
            fields: [Array(String), []],
        }));
    }
    _v_compute(source) {
        const n = source.get_length() ?? 0;
        const result = new Float64Array(n);
        for (const f of this.fields) {
            const column = source.data[f];
            if (column != null) {
                for (let i = 0, k = Math.min(n, column.length); i < k; i++) {
                    result[i] += column[i];
                }
            }
        }
        return result;
    }
}
Stack.__name__ = "Stack";
Stack.init_Stack();
//# sourceMappingURL=stack.js.map