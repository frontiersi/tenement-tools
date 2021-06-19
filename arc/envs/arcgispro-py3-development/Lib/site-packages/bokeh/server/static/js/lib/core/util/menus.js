import { div, classes, display, undisplay, empty, remove, Keys } from "../dom";
import { enumerate } from "./iterator";
import /*menus_css,*/ * as menus from "../../styles/menus.css";
export class ContextMenu {
    constructor(items, options = {}) {
        this.items = items;
        this.options = options;
        this.el = div();
        this._open = false;
        this._item_click = (i) => {
            this.items[i]?.handler();
            this.hide();
        };
        this._on_mousedown = (event) => {
            const { target } = event;
            if (target instanceof Node && this.el.contains(target))
                return;
            if (this.options.prevent_hide?.(event))
                return;
            this.hide();
        };
        this._on_keydown = (event) => {
            if (event.keyCode == Keys.Esc)
                this.hide();
        };
        this._on_blur = () => {
            this.hide();
        };
        undisplay(this.el);
    }
    get is_open() {
        return this._open;
    }
    get can_open() {
        return this.items.length != 0;
    }
    remove() {
        remove(this.el);
        this._unlisten();
    }
    _listen() {
        document.addEventListener("mousedown", this._on_mousedown);
        document.addEventListener("keydown", this._on_keydown);
        window.addEventListener("blur", this._on_blur);
    }
    _unlisten() {
        document.removeEventListener("mousedown", this._on_mousedown);
        document.removeEventListener("keydown", this._on_keydown);
        window.removeEventListener("blur", this._on_blur);
    }
    _position(at) {
        const parent_el = this.el.parentElement;
        if (parent_el != null) {
            const parent = parent_el.getBoundingClientRect();
            this.el.style.left = at.left != null ? `${at.left - parent.left}px` : "";
            this.el.style.top = at.top != null ? `${at.top - parent.top}px` : "";
            this.el.style.right = at.right != null ? `${parent.right - at.right}px` : "";
            this.el.style.bottom = at.bottom != null ? `${parent.bottom - at.bottom}px` : "";
        }
    }
    /*
    styles(): string[] {
      return [...super.styles(), menus_css]
    }
    */
    render() {
        empty(this.el, true);
        const orientation = this.options.orientation ?? "vertical";
        classes(this.el).add("bk-context-menu", `bk-${orientation}`);
        for (const [item, i] of enumerate(this.items)) {
            let el;
            if (item == null) {
                el = div({ class: menus.divider });
            }
            else if (item.if != null && !item.if()) {
                continue;
            }
            else {
                const icon = item.icon != null ? div({ class: ["bk-menu-icon", item.icon] }) : null;
                el = div({ class: item.active?.() ? "bk-active" : null, title: item.tooltip }, icon, item.label);
            }
            el.addEventListener("click", () => this._item_click(i));
            this.el.appendChild(el);
        }
    }
    show(at) {
        if (this.items.length == 0)
            return;
        if (!this._open) {
            this.render();
            if (this.el.children.length == 0)
                return;
            this._position(at ?? { left: 0, top: 0 });
            display(this.el);
            this._listen();
            this._open = true;
        }
    }
    hide() {
        if (this._open) {
            this._open = false;
            this._unlisten();
            undisplay(this.el);
        }
    }
    toggle(at) {
        this._open ? this.hide() : this.show(at);
    }
}
ContextMenu.__name__ = "ContextMenu";
//# sourceMappingURL=menus.js.map