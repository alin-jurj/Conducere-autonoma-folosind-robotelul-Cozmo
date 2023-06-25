import React, { useEffect, useContext } from 'react';
import { Text, View, StyleSheet, ImageBackground } from 'react-native';
import { BackgroundContext } from '../App';

function RobotCamera({ navigation }) {
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchData();
    }, 4000);

    return () => {
      clearInterval(intervalId);
    };
  }, []);

  const fetchData = () => {
    fetch('http://10.0.2.2:8000/request/')
      .then(response => response.json())
      .then(data => {
        if(data.length == 0){
          navigation.navigate('ClientScreen');
        }
      })
      .catch(error => {
        console.error(error);
      });
  };

  const backGroundContext = useContext(BackgroundContext);

  return (
    <View style={styles.container}>
      <ImageBackground source={{ uri: backGroundContext.background }} resizeMode="stretch" style={styles.image}>
      
      <View style={styles.container2}>
      
      <View style = {styles.bkgtext}>
          <Text style={styles.text}>Wait for Cozmo to reach destination</Text>
       </View>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
  },
  container2:{
  
    paddingTop: 350,
    paddingLeft: 50,
    width: 410,
    height: 840,
    paddingBottom:50,
   
  },
  bkgtext:{
    
    justifyContent: 'center',
    borderRadius:25,
    backgroundColor: 'rgba(54,86, 169, 0.8)',
    height:100,
    width: 300,
    
  },
  image: {
    flex: 1,
    justifyContent: 'center',
    width: 410,
    height: 840,
    paddingTop: 50,
    paddingLeft: 5,

  },
  text: {
    textAlign: 'center',
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    fontStyle: 'italic',
  },
  
});

export default RobotCamera;