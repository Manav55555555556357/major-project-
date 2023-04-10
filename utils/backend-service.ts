import axios, { AxiosPromise } from "axios";
import { PostalInfo } from "../interfaces"

export const getCity = async (pincode: number): Promise<AxiosPromise<PostalInfo[]>> => {
    return await axios.get(`https://api.postalpincode.in/pincode/${pincode}`)
}

export const getFoodFreshness = async (preparationTime: Date, imageUrl: string, pincode: number) => {
    const postalInfo = await getCity(pincode);
    const city = postalInfo.data[0].PostOffice[0].District;
    console.log({ "preparation_time": preparationTime, city, image_url: imageUrl })
    return await axios.post(`http://localhost:5000/predict`, {"preparation_time":preparationTime, city, image_url: imageUrl});
}