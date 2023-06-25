import { KeyboardAvoidingView,ImageBackground, Image, StyleSheet, Text, TextInput, Touchable, TouchableOpacity, View } from 'react-native'
import React, { useState, useContext } from 'react'
import Icon from "react-native-vector-icons/FontAwesome";
import * as ImagePicker from 'expo-image-picker';
import { BackgroundContext } from '../App';

function RegisterScreen  ({navigation}) {

  const backGroundContext = useContext(BackgroundContext);
    
    const [email, setEmail]  = useState('')
    const [username, setusername]  = useState('')
    const [password, setPassword]  = useState('')
    const [role, setRole] = useState('Passenger');
    const [pictureUrl, setPhoto] = useState(null);
    
    let handleRegisterPassenger = () =>{
      fetch("http://10.0.2.2:8000/user/signupPassenger",{
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body:JSON.stringify({
            username,
            email,
            password,
            pictureUrl
        })
      })
      .then(res=>res.text())
      .then( data=>{
        console.log(data);
        
      }).catch(error =>console.log(error))
      
    }

    let handleRegisterDriver = () =>{
      fetch("http://10.0.2.2:8000/user/signupDriver",{
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body:JSON.stringify({
            username,
            email,
            password,
            pictureUrl
        })
      })
      .then(res=>res.text())
      .then( data=>{
        data = JSON.parse(data);
        
        console.log(data.status);
        if(data.status == 'ok'){
          navigation.navigate('Login');
        }
        
      }).catch(error =>console.log(error))
      
    }

    const uploadImage = async () => {
      let result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.All,
          allowsEditing: true,
          aspect: [4, 3],
          quality: 1,
        });
    
        console.log(result);
    
        if (!result.canceled) {
          setPhoto(result.assets[0].uri);
        }
  }
  
  return (
   <KeyboardAvoidingView
    style={styles.container}
    behavior="padding">
      <ImageBackground source={{uri: backGroundContext.background}} resizeMode="stretch" style={[styles.image,styles.container]}>
        <View style = {styles.inputContainer}>
          <TextInput
            placeholder='Username'
            value = {username}
            onChangeText={text => setusername(text)}
          style = {styles.input}
          />
          <TextInput
            placeholder='Email'
            value = {email}
            onChangeText={text => setEmail(text)}
          style = {styles.input}
          />
          <TextInput
            placeholder='Password'
            value = {password}
            onChangeText={text => setPassword(text)}
          style = {styles.input}
          secureTextEntry
          />
          {role == 'Driver' ?
          <>
          {pictureUrl != null &&
            <View style={{backgroundColor: 'white', borderRadius: 20, marginTop: 5, flexDirection: 'row', alignItems: 'center', paddingVertical: 10, paddingHorizontal: 15}}>
              <Text style={{color: 'grey'}}>Chosen image: </Text>
              <Image source={{uri: pictureUrl}} style={{width: 100, height: 40, borderRadius: 5}}></Image>
            </View>
          }
            <TouchableOpacity style={styles.uploadButton}  onPress={() => uploadImage()}>
              <Icon name="upload" color={'grey'} size={25}></Icon>
              <Text style={{marginLeft: 5, color: 'grey'}}>{pictureUrl == null ? 'Upload driving license...' : 'Choose another image...'}</Text>
            </TouchableOpacity>
          </>
          :
          <>
          {pictureUrl != null &&
            <View style={{backgroundColor: 'white', borderRadius: 20, marginTop: 5, flexDirection: 'row', alignItems: 'center', paddingVertical: 10, paddingHorizontal: 15}}>
              <Text style={{color: 'grey'}}>Chosen image: </Text>
              <Image source={{uri: pictureUrl}} style={{width: 100, height: 40, borderRadius: 5}}></Image>
            </View>
          }
            <TouchableOpacity style={styles.uploadButton}  onPress={() => uploadImage()}>
              <Icon name="upload" color={'grey'} size={25}></Icon>
              <Text style={{marginLeft: 5, color: 'grey'}}>{pictureUrl == null ? 'Upload profile picture...' : 'Choose another image...'}</Text>
            </TouchableOpacity>
          </>
          
          }
        </View>

        {/* <View style={{flexDirection: 'row', marginTop: 30, width: 150, justifyContent: 'space-evenly'}}>
          <View style={{backgroundColor: role == 'Passenger' ? '#0782F9' : 'white', width: 50, height: 50, borderRadius: 50, alignItems: 'center', justifyContent: 'center'}}>
            <Icon name="street-view" color={role == 'Passenger' ? 'white' : '#0782F9'} size={30} onPress={() => setRole('Passenger')}></Icon>
          </View>
          <View style={{backgroundColor: role == 'Driver' ? '#0782F9' : 'white', width: 50, height: 50, borderRadius: 50, alignItems: 'center', justifyContent: 'center'}}>
            <Icon name="car" color={role == 'Driver' ? 'white' : '#0782F9'} size={25} onPress={() => setRole('Driver')}></Icon>
          </View>
        </View> */}

        <View style = {styles.buttonContainer}>
          <TouchableOpacity
            onPress={role == 'Passenger' ? handleRegisterPassenger : handleRegisterDriver}
            style= {styles.button}
          >
            <Text style = {styles.buttonText}>Register</Text>  
          </TouchableOpacity>
        </View>
      </ImageBackground>
    </KeyboardAvoidingView>
  )
}

export default RegisterScreen

const styles = StyleSheet.create({
    container:{
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    inputContainer: {
        width: '80%'
    },
    input:{
        backgroundColor: 'white',
        paddingHorizontal:15,
        paddingVertical:10,
        borderRadius:10,
        marginTop:5,
    },
    uploadButton: {
        backgroundColor: 'white',
        paddingHorizontal:15,
        paddingVertical:10,
        borderRadius:10,
        marginTop:5,
        flexDirection: 'row',
        alignItems: 'flex-end',
        justifyContent: 'center'
    },
    buttonContainer:{
        width:'60%',
        justifyContent:'center',
        alignItems: 'center',
        marginTop: 40,
    },
    button: {
      backgroundColor: '#0782F9',
      width: '100%',
      padding: 15,
      borderRadius: 10,
      alignItems: 'center',
    },
    buttonOutline:{
      backgroundColor:'white',
      marginTop:5,
      borderColor: '#0782F9',
      borderWidth: 2,
    },
    buttonText:{
      color: 'white',
      fontWeight: '700',
      fontSize: 16,
    },
    buttonOutlineText:{
      color: '#0782F9',
      fontWeight: '700',
      fontSize: 16,
    },
    image: {
      flex: 1,
      justifyContent: "center",
      width:410,
      height:900,
    }
})