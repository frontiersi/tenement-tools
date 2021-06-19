import { input } from "../../core/dom";
import { Widget, WidgetView } from "./widget";
export class FileInputView extends WidgetView {
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.change, () => this.render());
    }
    render() {
        const { multiple, accept, disabled, width } = this.model;
        if (this.dialog_el == null) {
            this.dialog_el = input({ type: "file", multiple });
            this.dialog_el.onchange = () => {
                const { files } = this.dialog_el;
                if (files != null) {
                    this.load_files(files);
                }
            };
            this.el.appendChild(this.dialog_el);
        }
        if (accept != null && accept != "") {
            this.dialog_el.accept = accept;
        }
        this.dialog_el.style.width = `${width}px`;
        this.dialog_el.disabled = disabled;
    }
    async load_files(files) {
        const values = [];
        const filenames = [];
        const mime_types = [];
        for (const file of files) {
            const data_url = await this._read_file(file);
            const [, mime_type = "", , value = ""] = data_url.split(/[:;,]/, 4);
            values.push(value);
            filenames.push(file.name);
            mime_types.push(mime_type);
        }
        if (this.model.multiple) {
            this.model.value = values;
            this.model.filename = filenames;
            this.model.mime_type = mime_types;
        }
        else {
            this.model.value = values[0];
            this.model.filename = filenames[0];
            this.model.mime_type = mime_types[0];
        }
    }
    _read_file(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const { result } = reader;
                if (result != null) {
                    resolve(result);
                }
                else {
                    reject(reader.error ?? new Error(`unable to read '${file.name}'`));
                }
            };
            reader.readAsDataURL(file);
        });
    }
}
FileInputView.__name__ = "FileInputView";
export class FileInput extends Widget {
    constructor(attrs) {
        super(attrs);
    }
    static init_FileInput() {
        this.prototype.default_view = FileInputView;
        this.define(({ Boolean, String, Array, Or }) => ({
            value: [Or(String, Array(String)), ""],
            mime_type: [Or(String, Array(String)), ""],
            filename: [Or(String, Array(String)), ""],
            accept: [String, ""],
            multiple: [Boolean, false],
        }));
    }
}
FileInput.__name__ = "FileInput";
FileInput.init_FileInput();
//# sourceMappingURL=file_input.js.map