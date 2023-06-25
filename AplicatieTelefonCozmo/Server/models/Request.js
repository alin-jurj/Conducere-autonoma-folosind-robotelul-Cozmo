const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const userSchema = new Schema({
    passengerUsername: {
        type: String,
        required: true
    },
    destination:{
        type: String,
        required: true
    },
    passengerPicture: {
        type: String,
        required: false
    },
    
    distance: {
        type: String,
        required: false
    },
    
});

module.exports = mongoose.model('Request', userSchema)