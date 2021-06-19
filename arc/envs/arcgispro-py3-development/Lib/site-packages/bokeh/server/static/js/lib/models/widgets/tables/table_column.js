import { StringFormatter } from "./cell_formatters";
import { StringEditor } from "./cell_editors";
import { uniqueId } from "../../../core/util/string";
import { Sort } from "../../../core/enums";
import { Model } from "../../../model";
export class TableColumn extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_TableColumn() {
        this.define(({ Boolean, Number, String, Nullable, Ref }) => ({
            field: [String],
            title: [Nullable(String), null],
            width: [Number, 300],
            formatter: [Ref(StringFormatter), () => new StringFormatter()],
            editor: [Ref(StringEditor), () => new StringEditor()],
            sortable: [Boolean, true],
            default_sort: [Sort, "ascending"],
        }));
    }
    toColumn() {
        return {
            id: uniqueId(),
            field: this.field,
            name: this.title ?? this.field,
            width: this.width,
            formatter: this.formatter != null ? this.formatter.doFormat.bind(this.formatter) : undefined,
            model: this.editor,
            editor: this.editor.default_view,
            sortable: this.sortable,
            defaultSortAsc: this.default_sort == "ascending",
        };
    }
}
TableColumn.__name__ = "TableColumn";
TableColumn.init_TableColumn();
//# sourceMappingURL=table_column.js.map