const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const userSchema = new Schema({
    username:{
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true
    },
    password: {
        type: String,
        required: true
    },
    pictureUrl: {
        type: String,
        required: false
    },
    token: {
        type: String,
        required: false
    }
    // avatar: Buffer,
});
//export default mongoose.model("Passanger",userSchema);
module.exports = mongoose.model('Passanger', userSchema)
