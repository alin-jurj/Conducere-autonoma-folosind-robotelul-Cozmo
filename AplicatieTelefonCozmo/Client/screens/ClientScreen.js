import React, { useState, useEffect } from 'react'
import { Text, View, ImageBackground, StyleSheet, TextInput, FlatList, Image, TouchableOpacity, Keyboard, Pressable } from 'react-native'
import {LoginContext} from '../App'
import { useContext } from 'react';
import { BackgroundContext } from '../App';



function ClientScreen({navigation}) {
  const context = useContext(LoginContext);

  const backGroundContext = useContext(BackgroundContext);


  const backgrounds = [
    {
        id: 1,
        url: 'https://img.freepik.com/free-vector/large-school-building-scene_1308-32058.jpg?w=2000'
       
      },
    {
        id: 2,
        url: 'https://img.freepik.com/free-vector/hand-drawn-church-building-illustration_23-2149453119.jpg?w=2000'
    },
    {
        id: 3,
        url: 'https://external-preview.redd.it/police-academy-the-animated-series-v0-DtoZhMb8jVcdgp8gxQ18g2zT92157S71363isAC7p0c.png?format=pjpg&auto=webp&s=ce3dbd8501c7f4ffb14ddf03b001b8afea71f9d6'
    }
]

const passengerUsername = context.loginDetails.username;

const handleRequest = function(id) {
  let destination = '';
  
  if (id === 1) {
    destination = 'School';
  } else if (id === 2) {
    destination = 'Church';
  } else if (id === 3) {
    destination = 'Police';
  }
 
  fetch('http://10.0.2.2:8000/request/add', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      passengerUsername,
      destination,
    }),
  });
  navigation.navigate('RobotCamera')
};
const BackGroundContainer = ({item}) => (
  <View style={{marginLeft: 30, marginRight: 20, marginTop: 110}}>
      <Image source={{uri: item.url}} style={{width: 150, height: 200, borderTopLeftRadius: 20, borderTopRightRadius: 20}}></Image>
      <TouchableOpacity style={styles.button} onPress={() => handleRequest(item.id)} >
          <Text style={{color: 'white', fontSize: 12}}>Choose destination</Text>
      </TouchableOpacity>
  </View>
)
  return (
    <View style={styles.container}>
        <ImageBackground source={{uri: backGroundContext.background}} resizeMode="stretch" style={styles.image}>
        <FlatList 
                data={backgrounds}
                renderItem={({item}) => <BackGroundContainer item={item} />}
                keyExtractor={item => item.id} 
                numColumns={2}
            />
        </ImageBackground>
  </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center'
  },
  image: {
    flex: 1,
    justifyContent: "center",
    width:410,
    height:820,
    paddingTop: 50,
    paddingLeft: 5
  },
  button: {
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 8,
      paddingHorizontal: 18,
      borderBottomRightRadius: 20,
      borderBottomLeftRadius: 20,
      backgroundColor: '#01135d',
      width: 150
  },

  });

export default ClientScreen