export interface User {
  email: string;
  uid: string;
  userType: string;
  userData: RestaurantUserData | NGOUserData;
}

export interface RestaurantUserData {
  address: string;
  gstNumber: string;
  latitude: string;
  longitude: string;
  name: string;
  pinCode: string;
}

export interface NGOUserData {
  Address: string;
  Designation: string;
  District: string;
  'Email id': string;
  Mobile: number;
  'NGO Head Name': string;
  'NGO Name': string;
  SNo: number;
  State: string;
}

export interface RestaurantUser {
  email: string;
  uid: string;
  userData: {
    address: string;
    gstNumber: string;
    latitude: string;
    longitude: string;
    name: string;
    pinCode: string;
  };
  donations?: Donation[];
  userType: string;
}

export interface NGOUser {
  email: string;
  uid: string;
  userData: {
    Address: string;
    Designation: string;
    District: string;
    'Email id': string;
    Mobile: number;
    'NGO Head Name': string;
    'NGO Name': string;
    SNo: number;
    State: string;
  };
  userType: string;
}

export interface Donation {
  id?: string;
  foodName: string;
  imageUrl: string;
  quantity: number;
  restaurantId: string;
  status: string;
  timestamp: Date;
}

export interface PostalInfo {
  Message: string;
  Status: string;
  PostOffice: PostOffice[];
}

export interface PostOffice {
  Name: string;
  Description: string | null;
  BranchType: string;
  DeliveryStatus: string;
  Circle: string;
  District: string;
  Division: string;
  Region: string;
  Block: string;
  State: string;
  Country: string;
  Pincode: string;
}