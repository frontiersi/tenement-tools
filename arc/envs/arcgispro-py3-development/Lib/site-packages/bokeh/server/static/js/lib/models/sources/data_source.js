import { Model } from "../../model";
import { Selection } from "../selections/selection";
export class DataSource extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_DataSource() {
        this.define(({ Ref }) => ({
            selected: [Ref(Selection), () => new Selection()],
        }));
    }
}
DataSource.__name__ = "DataSource";
DataSource.init_DataSource();
//# sourceMappingURL=data_source.js.map