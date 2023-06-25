const Passanger = require('../models/Passanger');

const searchEmail = async(email) => {
    let emailExist = await Passanger.findOne({email: email});

    if(emailExist)
        return new Error('Email already exists');
    return null;
};
module.exports.searchEmail = searchEmail;