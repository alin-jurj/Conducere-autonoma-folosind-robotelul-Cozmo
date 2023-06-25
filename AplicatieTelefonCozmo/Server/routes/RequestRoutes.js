const express = require('express');
const {addRequest,updateRequest, getRequest,deleteRequest} = require('../controllers/RequestController');
const router = express.Router();

router.post('/add',addRequest);
router.put('/update', updateRequest);
router.get('/', getRequest);
router.delete('/delete',deleteRequest)
module.exports = router;