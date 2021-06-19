import { Model } from "../../model";
import { keys, values } from "../../core/util/object";
import { use_strict } from "../../core/util/string";
import { isIterable } from "../../core/util/types";
import { Indices, GeneratorFunction } from "../../core/types";
export class LabelingPolicy extends Model {
    constructor(attrs) {
        super(attrs);
    }
}
LabelingPolicy.__name__ = "LabelingPolicy";
export class AllLabels extends LabelingPolicy {
    constructor(attrs) {
        super(attrs);
    }
    filter(indices, _bboxes, _distance) {
        return indices;
    }
}
AllLabels.__name__ = "AllLabels";
export class NoOverlap extends LabelingPolicy {
    constructor(attrs) {
        super(attrs);
    }
    static init_NoOverlap() {
        this.define(({ Number }) => ({
            min_distance: [Number, 5],
        }));
    }
    filter(indices, _bboxes, distance) {
        const { min_distance } = this;
        let k = null;
        for (const i of indices) {
            if (k != null && distance(k, i) < min_distance)
                indices.unset(i);
            else
                k = i;
        }
        return indices;
    }
}
NoOverlap.__name__ = "NoOverlap";
NoOverlap.init_NoOverlap();
export class CustomLabelingPolicy extends LabelingPolicy {
    constructor(attrs) {
        super(attrs);
    }
    static init_CustomLabelingPolicy() {
        this.define(({ Unknown, String, Dict }) => ({
            args: [Dict(Unknown), {}],
            code: [String, ""],
        }));
    }
    get names() {
        return keys(this.args);
    }
    get values() {
        return values(this.args);
    }
    get func() {
        const code = use_strict(this.code);
        return new GeneratorFunction("indices", "bboxes", "distance", ...this.names, code);
    }
    filter(indices, bboxes, distance) {
        const obj = Object.create(null);
        const generator = this.func.call(obj, indices, bboxes, distance, ...this.values);
        let result = generator.next();
        if (result.done && result.value !== undefined) {
            const { value } = result;
            if (value instanceof Indices)
                return value;
            else if (value === undefined)
                return indices;
            else if (isIterable(value))
                return Indices.from_indices(indices.size, value);
            else
                return Indices.all_unset(indices.size);
        }
        else {
            const array = [];
            do {
                array.push(result.value);
                result = generator.next();
            } while (!result.done);
            return Indices.from_indices(indices.size, array);
        }
    }
}
CustomLabelingPolicy.__name__ = "CustomLabelingPolicy";
CustomLabelingPolicy.init_CustomLabelingPolicy();
//# sourceMappingURL=labeling.js.map