import { Widget } from "../widget";
import { ColumnDataSource } from "../../sources/column_data_source";
import { CDSView } from "../../sources/cds_view";
export class TableWidget extends Widget {
    constructor(attrs) {
        super(attrs);
    }
    static init_TableWidget() {
        this.define(({ Ref }) => ({
            source: [Ref(ColumnDataSource), () => new ColumnDataSource()],
            view: [Ref(CDSView), () => new CDSView()],
        }));
    }
    initialize() {
        super.initialize();
        if (this.view.source == null) {
            this.view.source = this.source;
            this.view.compute_indices();
        }
    }
}
TableWidget.__name__ = "TableWidget";
TableWidget.init_TableWidget();
//# sourceMappingURL=table_widget.js.map