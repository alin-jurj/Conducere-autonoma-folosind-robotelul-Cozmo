const express = require('express');
const {loginUser, getPassangers,signupPassenger} = require('../controllers/userController');
const router = express.Router();

router.get('/',getPassangers)
router.post('/signin',loginUser);
router.post('/signupPassenger', signupPassenger);

//router.post('/register',postPassanger)
module.exports = router;