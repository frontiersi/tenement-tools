import { Model } from "../../model";
import { TextureRepetition } from "../../core/enums";
export class Texture extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_Texture() {
        this.define(() => ({
            repetition: [TextureRepetition, "repeat"],
        }));
    }
}
Texture.__name__ = "Texture";
Texture.init_Texture();
//# sourceMappingURL=texture.js.map