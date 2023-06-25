import { View, Text, ImageBackground, StyleSheet, FlatList, Image, Touchable, TouchableOpacity } from 'react-native'
import React, {useContext} from 'react'
import { BackgroundContext } from '../App'

const BackgroundScreen = () => {
    const backgroundContext = useContext(BackgroundContext);

    const backgrounds = [
        {
            id: 1,
            url: 'https://helios-i.mashable.com/imagery/articles/06zoscMHTZxU5KEFx8SRyDg/hero-image.fill.size_1200x1200.v1630023012.jpg'
        },
        {
            id: 2,
            url: 'https://media.istockphoto.com/id/1185361371/vector/travel-to-the-forest-by-car-road-trip-time-to-travel-tourism-summer-holiday-different-types.jpg?s=170667a&w=0&k=20&c=CMkuWmmnkaMbI9-XC91MPxwDWGy4LY9CgKT9CjOn1Uc='
        },
        {
            id: 3,
            url: 'https://freedesignfile.com/upload/2017/06/Go-vacation-travel-with-car-vector.jpg'
        },
        {
            id: 4,
            url: 'https://m.media-amazon.com/images/I/B1+BUJw5c0S.png'
        }
    ]

    const BackGroundContainer = ({item}) => (
        <View style={{marginLeft: 20, marginRight: 20, marginTop: 110}}>
            <Image source={{uri: item.url}} style={{width: 150, height: 200, borderTopLeftRadius: 20, borderTopRightRadius: 20}}></Image>
            <TouchableOpacity style={styles.button} onPress={() => backgroundContext.setBackground(item.url)}>
                <Text style={{color: 'white', fontSize: 12}}>Choose</Text>
            </TouchableOpacity>
        </View>
    )

  return (
        <ImageBackground source={{ uri: backgroundContext.background}} style={styles.image} resizeMode="stretch">
            <FlatList 
                data={backgrounds}
                renderItem={({item}) => <BackGroundContainer item={item} />}
                keyExtractor={item => item.id} 
                numColumns={2}
            />
        </ImageBackground>
  )
}

export default BackgroundScreen

const styles = StyleSheet.create({
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