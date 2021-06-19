import { Model } from "../../model";
export class Range extends Model {
    constructor(attrs) {
        super(attrs);
        this.have_updated_interactively = false;
    }
    static init_Range() {
        this.define(({ Number, Tuple, Or, Auto, Nullable }) => ({
            bounds: [Nullable(Or(Tuple(Nullable(Number), Nullable(Number)), Auto)), null],
            min_interval: [Nullable(Number), null],
            max_interval: [Nullable(Number), null],
        }));
        this.internal(({ Array, AnyRef }) => ({
            plots: [Array(AnyRef()), []],
        }));
    }
    get is_reversed() {
        return this.start > this.end;
    }
    get is_valid() {
        return isFinite(this.min) && isFinite(this.max);
    }
}
Range.__name__ = "Range";
Range.init_Range();
//# sourceMappingURL=range.js.map